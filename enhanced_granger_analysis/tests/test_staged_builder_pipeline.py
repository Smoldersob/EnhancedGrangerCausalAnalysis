from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ..api import orchestrator as orchestrator_module
from ..api.builder import MultiTaskGrangerBuilder
from ..core.exceptions import DataValidationError
from ..preprocessing.prepare_data import PreparedData, VariableRoles


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=float).copy()


class _Constraint:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)


class _RecordingModel:
    def __init__(self, n_features, n_outputs, callbacks=None, **config):
        self.weights = [
            np.arange(n_features * n_outputs, dtype=float).reshape(n_features, n_outputs),
            np.zeros(n_outputs, dtype=float),
        ]
        self.callbacks = callbacks or []
        self.omitted = []
        self.reset_calls = []
        self.fit_calls = 0
        self.initialized = None

    def initialize(self, X, targets=None):
        self.initialized = (np.asarray(X).copy(), np.asarray(targets).copy())

    def get_weights(self):
        return [weight.copy() for weight in self.weights]

    def set_weights(self, weights):
        self.weights = [np.asarray(weight, dtype=float).copy() for weight in weights]

    def omit_variables(self, variable_indices):
        self.omitted = list(variable_indices)

    def reset_callbacks(self, run_name, callback_updates=None):
        self.reset_calls.append((run_name, callback_updates))

    def fit(self):
        self.fit_calls += 1
        for index in self.omitted:
            self.weights[0][index, :] = 0.0

    def predict(self, X):
        return np.zeros((len(X), self.weights[1].shape[0]), dtype=float)


class _RecordingStrategy:
    def __init__(self):
        self.model = None
        self.constraint_calls = []

    def build_constraint_from_relations(self, **kwargs):
        self.constraint_calls.append(kwargs)
        return _Constraint(kwargs["base_mask"])

    def build_regularizer(self, spec):
        return ("regularizer", dict(spec))

    def build_model(self, **kwargs):
        self.model = _RecordingModel(**kwargs)
        return self.model


def _prepared_data() -> PreparedData:
    prepared = PreparedData(
        X_scaled=np.arange(24, dtype=float).reshape(6, 4),
        y_scaled=np.arange(6, dtype=float).reshape(6, 1),
        X_raw=np.arange(24, dtype=float).reshape(6, 4),
        y_raw=np.arange(6, dtype=float).reshape(6, 1),
        x_scaler=_IdentityScaler(),
        y_scaler=_IdentityScaler(),
        base_mask=np.ones((1, 4), dtype=float),
        requested_roles=VariableRoles(("effect",), ("context", "x", "z"), ("z", "x")),
        effective_roles=VariableRoles(("effect",), ("context", "x", "z"), ("z", "x")),
        cause_slices={
            "context": slice(0, 1),
            "x": slice(1, 3),
            "z": slice(3, 4),
        },
    )
    prepared.stationarity_transformer = SimpleNamespace()
    return prepared


def _configured_builder(monkeypatch):
    strategy = _RecordingStrategy()
    monkeypatch.setattr(
        "enhanced_granger_analysis.api.builder.BackendFactory.get_strategy",
        staticmethod(lambda backend: strategy),
    )
    builder = (
        MultiTaskGrangerBuilder(backend="dummy")
        .data(pd.DataFrame({"context": [1], "x": [2], "z": [3], "effect": [4]}))
        .variables(
            causes=["context", "x", "z"],
            effects=["effect"],
            tested_causes=["z", "x"],
        )
        .regularization(regularizer_spec={"type": "l1", "strength": 0.1})
    )
    builder.prepare_data(_prepared_data())
    builder.prepare_components()
    builder.build_model({"epochs": 2})
    return builder, strategy


def test_stages_require_their_prerequisites():
    builder = MultiTaskGrangerBuilder()

    with pytest.raises(DataValidationError, match="prepare_data"):
        builder.prepare_backend()
    with pytest.raises(DataValidationError, match="requires data"):
        builder.prepare_data()


def test_builder_reuses_one_model_and_restricted_runs_start_from_base_weights(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)
    model = strategy.model

    builder.fit_base()
    base_weights = builder.results.base_weights.copy()
    builder.fit_restricted_models()

    assert strategy.model is model
    assert model.fit_calls == 3
    assert model.reset_calls[0][0] == "base_model"
    assert [call[0] for call in model.reset_calls[1:]] == [
        "reference_cause_z",
        "reference_cause_x",
    ]
    assert model.omitted == [1, 2]
    assert np.array_equal(builder.results.base_weights, base_weights)
    assert np.all(builder.results.ref_weights["z"][3, :] == 0.0)
    assert np.all(builder.results.ref_weights["x"][1:3, :] == 0.0)


def test_restricted_blocks_are_selected_by_cause_name_not_test_order(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)

    builder.fit_base()
    builder.fit_restricted_models()

    # tested_causes is [z, x], while x occupies the middle feature block.
    assert strategy.model.reset_calls[-1][0] == "reference_cause_x"
    assert strategy.model.omitted == [1, 2]
    assert builder.results.effects == ["effect"]
    assert builder.results.causes == ["z", "x"]


def test_weights_do_not_change_the_input_mask_and_snapshots_are_independent(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)
    builder.fit_base()
    base_before = builder.results.base_weights.copy()

    builder.fit_restricted("z")

    assert np.array_equal(builder.results.base_weights, base_before)
    assert np.array_equal(strategy.model.weights[0][1:3], base_before[1:3])
    assert strategy.model.omitted == [3]


def test_reset_callbacks_is_delegated_with_updates(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)

    builder.reset_callbacks("manual_run", {"EarlyStopping": {"patience": 2}})

    assert strategy.model.reset_calls == [
        ("manual_run", {"EarlyStopping": {"patience": 2}})
    ]


def test_orchestrator_runs_the_standard_builder_sequence(monkeypatch):
    calls = []

    class FakeBuilder:
        def __init__(self, backend=None, reuse_data=False):
            calls.append(("init", backend, reuse_data))

        def data(self, value):
            calls.append("data")
            return self

        def variables(self, *values):
            calls.append("variables")
            return self

        def relations(self, value):
            calls.append("relations")
            return self

        def lag(self, *values):
            calls.append("lag")
            return self

        def stationarity(self, value):
            calls.append("stationarity")
            return self

        def prepare_data(self, value):
            calls.append("prepare_data")

        def prepare_components(self):
            calls.append("prepare_components")

        def build_model(self):
            calls.append("build_model")

        def initialize_weights(self):
            calls.append("initialize_weights")

        def fit_base(self):
            calls.append("fit_base")

        def fit_restricted_models(self):
            calls.append("fit_restricted_models")

        def get_results(self):
            calls.append("get_results")
            return "results"

        _fit_kwargs = {}

    monkeypatch.setattr(orchestrator_module, "MultiTaskGrangerBuilder", FakeBuilder)
    result = orchestrator_module.MultiTaskGrangerAPI("dummy", reuse_data=True).fit(
        pd.DataFrame({"x": [1], "y": [2]}),
        causes=["x"],
        effects=["y"],
        tested_causes=["x"],
    )

    assert result == "results"
    assert calls[-6:] == [
        "prepare_data",
        "prepare_components",
        "build_model",
        "initialize_weights",
        "fit_base",
        "fit_restricted_models",
    ] or calls[-7:] == [
        "prepare_data",
        "prepare_components",
        "build_model",
        "initialize_weights",
        "fit_base",
        "fit_restricted_models",
        "get_results",
    ]
