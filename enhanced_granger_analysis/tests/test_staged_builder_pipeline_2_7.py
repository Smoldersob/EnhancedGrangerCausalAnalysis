from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enhanced_granger_analysis.core.models.training_result import TrainingResult
from enhanced_granger_analysis.api import orchestrator as orchestrator_module
from enhanced_granger_analysis.api.builder import MultiTaskGrangerBuilder
from enhanced_granger_analysis.core.exceptions import DataValidationError
from enhanced_granger_analysis.preprocessing.prepare_data import PreparedData, VariableRoles


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values, dtype=float).copy()


class _Constraint:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)


class _RecordingModel:
    def __init__(
        self,
        *,
        input_spec,
        optimizer,
        loss,
        callbacks=None,
        regularizer=None,
        constraint=None,
        **config,
    ):
        self.input_spec = input_spec
        self.optimizer = optimizer
        self.loss = loss
        self.callbacks = list(callbacks or [])
        self.regularizer = regularizer
        self.constraint = constraint
        self.config = dict(config)

        self.weights = [
            np.arange(
                input_spec.n_features * input_spec.n_outputs,
                dtype=float,
            ).reshape(
                input_spec.n_features,
                input_spec.n_outputs,
            ),
            np.zeros(input_spec.n_outputs, dtype=float),
        ]

        self.omitted = []
        self.reset_calls = []
        self.component_reset_calls = []
        self.fit_calls = 0
        self.fit_inputs = []
        self.fit_start_weights = []

    def get_weights(self):
        return [weight.copy() for weight in self.weights]

    def set_weights(self, weights):
        self.weights = [
            np.asarray(weight, dtype=float).copy()
            for weight in weights
        ]

    def omit_variables(self, variable_indices):
        self.omitted = list(variable_indices)

    def reset_components(
        self,
        *,
        reset_optimizer=True,
        reset_callbacks=True,
        run_name=None,
        callback_updates=None,
    ):
        self.component_reset_calls.append({
            "reset_optimizer": reset_optimizer,
            "reset_callbacks": reset_callbacks,
            "run_name": run_name,
            "callback_updates": callback_updates,
        })

        if reset_callbacks:
            self.reset_calls.append((run_name, callback_updates))

    def fit(
        self,
        X,
        y,
        *,
        collect_history=False,
        reset_components=True,
        run_name=None,
        callback_updates=None,
    ):
        if reset_components:
            self.reset_components(
                run_name=run_name,
                callback_updates=callback_updates,
            )

        self.fit_calls += 1
        self.fit_inputs.append((
            np.asarray(X).copy(),
            np.asarray(y).copy(),
        ))
        self.fit_start_weights.append(self.get_weights())

        return TrainingResult()

    def predict(self, X):
        inputs = np.asarray(X, dtype=float).copy()
        inputs[:, self.omitted] = 0.0
        return inputs @ self.weights[0] + self.weights[1]


class _RecordingStrategy:
    def __init__(self):
        self.model = None
        self.model_calls = []
        self.constraint_calls = []
        self.regularizer_calls = []
        self.optimizer_calls = []
        self.loss_calls = []
        self.callback_calls = []

    def build_constraint(self, spec=None, *, data=None):
        self.constraint_calls.append((spec, data))
        return _Constraint(data.mask)

    def build_regularizer(self, spec=None, *, data=None):
        self.regularizer_calls.append((spec, data))
        return ("regularizer", spec, data)

    def build_optimizer(self, spec=None, **context):
        assert not context
        self.optimizer_calls.append(spec)
        return ("optimizer", spec)

    def build_loss(self, spec=None):
        self.loss_calls.append(spec)
        return ("loss", spec)

    def build_callbacks(self, specs=None):
        self.callback_calls.append(specs)
        return [] if specs is None else list(specs)

    def build_model(self, spec=None, *, components=None, config=None):
        components = dict(components or {})
        config = dict(config or {})

        self.model_calls.append({
            "spec": spec,
            "components": components,
            "config": config,
        })

        self.model = _RecordingModel(
            **config,
            **components,
        )
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
        feature_lags=np.arange(4, dtype=np.int64),
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
        .regularization(regularizer_spec={"type": "l1", "params": {"l1": 0.1}})
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


def test_builder_reuses_one_model_and_restricted_runs_start_from_base_weights(
    monkeypatch,
):
    builder, strategy = _configured_builder(monkeypatch)
    model = strategy.model

    builder.fit_base()
    base_weights = [
        weight.copy()
        for weight in model.get_weights()
    ]
    base_snapshot_weights = builder.results.base_weights.copy()

    # Deliberately disturb the active model after saving the base weights.
    model.set_weights([
        np.full_like(weight, -100.0)
        for weight in base_weights
    ])

    builder.fit_restricted_models()

    assert strategy.model is model
    assert len(strategy.model_calls) == 1
    assert model.fit_calls == 3

    assert [call[0] for call in model.reset_calls] == [
        "base_model",
        "reference_cause_z",
        "reference_cause_x",
    ]

    assert all(
        call["reset_optimizer"] and call["reset_callbacks"]
        for call in model.component_reset_calls
    )

    for start_weights in model.fit_start_weights[1:]:
        for actual, expected in zip(start_weights, base_weights):
            np.testing.assert_array_equal(actual, expected)

    assert model.omitted == [1, 2]

    np.testing.assert_array_equal(
        builder.results.base_weights,
        base_snapshot_weights,
    )

    prepared = builder._prepared_data

    for cause in ("z", "x"):
        masked_X = prepared.X_scaled.copy()
        masked_X[:, prepared.cause_slices[cause]] = 0.0

        expected = prepared.y_scaler.inverse_transform(
            masked_X @ base_weights[0] + base_weights[1]
        )

        np.testing.assert_allclose(
            builder.results.ref_predictions[cause],
            expected,
        )

        # Input masking does not require coefficient zeroing.
        np.testing.assert_array_equal(
            builder.results.ref_weights[cause],
            base_snapshot_weights,
        )


def test_restricted_blocks_are_selected_by_cause_name_not_test_order(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)

    builder.fit_base()
    builder.fit_restricted_models()

    # tested_causes is [z, x], while x occupies the middle feature block.
    assert strategy.model.reset_calls[-1][0] == "reference_cause_x"
    assert strategy.model.omitted == [1, 2]
    assert builder.results.effects == ["effect"]
    assert builder.results.causes == ["z", "x"]


def test_weights_do_not_change_the_input_mask_and_snapshots_are_independent(
    monkeypatch,
):
    builder, strategy = _configured_builder(monkeypatch)

    prepared_mask_before = builder._prepared_data.base_mask.copy()
    results_mask_before = builder.results.mask.copy()

    builder.fit_base()
    base_before = builder.results.base_weights.copy()

    builder.fit_restricted("z")
    reference_before = builder.results.ref_weights["z"].copy()

    assert strategy.model.omitted == [3]

    np.testing.assert_array_equal(
        builder._prepared_data.base_mask,
        prepared_mask_before,
    )
    np.testing.assert_array_equal(
        builder.results.mask,
        results_mask_before,
    )
    np.testing.assert_array_equal(
        builder.results.base_weights,
        base_before,
    )

    strategy.model.set_weights([
        np.full_like(weight, -999.0)
        for weight in strategy.model.get_weights()
    ])

    np.testing.assert_array_equal(
        builder.results.base_weights,
        base_before,
    )
    np.testing.assert_array_equal(
        builder.results.ref_weights["z"],
        reference_before,
    )

    assert strategy.model.omitted == [3]


def test_reset_callbacks_is_delegated_with_updates(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)

    updates = {"EarlyStopping": {"patience": 2}}
    builder.reset_callbacks("manual_run", updates)

    assert strategy.model.reset_calls == [
        ("manual_run", updates)
    ]

    assert strategy.model.component_reset_calls == [{
        "reset_optimizer": False,
        "reset_callbacks": True,
        "run_name": "manual_run",
        "callback_updates": updates,
    }]

def test_builder_passes_input_spec_and_ready_components(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)

    call = strategy.model_calls[0]
    config = call["config"]
    components = call["components"]
    input_spec = config["input_spec"]

    assert call["spec"] is None

    assert input_spec.n_features == 4
    assert input_spec.n_outputs == 1
    assert input_spec.causes == ("context", "x", "z")
    assert input_spec.effects == ("effect",)
    assert dict(input_spec.cause_slices) == {
        "context": slice(0, 1),
        "x": slice(1, 3),
        "z": slice(3, 4),
    }
    assert input_spec.feature_lags == (0, 1, 2, 3)

    assert config["epochs"] == 2
    assert "n_features" not in config
    assert "n_outputs" not in config

    assert set(components) == {
        "optimizer",
        "loss",
        "callbacks",
        "regularizer",
        "constraint",
    }

    assert components["optimizer"] == ("optimizer", None)
    assert components["loss"] == ("loss", None)
    assert components["callbacks"] == []
    assert components["constraint"] is builder._constraint
    assert components["regularizer"] is builder._regularizer

def test_each_fit_receives_the_same_prepared_observations(monkeypatch):
    builder, strategy = _configured_builder(monkeypatch)
    prepared = builder._prepared_data

    X_before = prepared.X_scaled.copy()
    y_before = prepared.y_scaled.copy()

    builder.fit_base()
    builder.fit_restricted_models()

    assert len(strategy.model.fit_inputs) == 3

    for X, y in strategy.model.fit_inputs:
        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(y, y_before)

    np.testing.assert_array_equal(prepared.X_scaled, X_before)
    np.testing.assert_array_equal(prepared.y_scaled, y_before)