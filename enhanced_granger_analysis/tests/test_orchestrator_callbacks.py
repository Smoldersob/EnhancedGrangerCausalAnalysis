import traceback

import numpy as np
import pandas as pd

from ..api import orchestrator as orchestrator_module
from ..api.orchestrator import MultiTaskGrangerAPI
from ..core.lag_config import LagConfiguration
from ..preprocessing.scaling import _BaseScaler
from ..preprocessing.stationarity import StationarityTransformer


from unittest import SkipTest


class _DummyCallback:
    def __init__(self, run_name: str = "template"):
        self.run_name = run_name

    def clone_for_run(self, run_name: str):
        return _DummyCallback(run_name=run_name)


class _DummyModel:
    seen_run_names = []

    def __init__(self, callbacks=None, n_features=1, n_outputs=1):
        self.callbacks = callbacks or []
        self._n_features = n_features
        self._n_outputs = n_outputs
        self._weights = [np.ones((n_features, n_outputs), dtype=np.float64)]
        self._X = None
        self._y = None

    def initialize(self, X, targets=None):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(targets, dtype=np.float64)

    def set_weights(self, weights):
        self._weights = weights

    def omit_variables(self, variable_indices):
        _ = variable_indices

    def fit(self):
        for cb in self.callbacks:
            _DummyModel.seen_run_names.append(getattr(cb, "run_name", "unknown"))
        return {"test_statistic": 0.0}

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return np.zeros((X_arr.shape[0], self._n_outputs), dtype=np.float64)

    def get_weights(self):
        return self._weights


class _DummyStrategy:
    def build_model(self, n_features, n_outputs, regularizer=None, constraint=None, scaler=None, **config):
        _ = regularizer, constraint, scaler
        return _DummyModel(
            callbacks=config.get("callbacks", None),
            n_features=n_features,
            n_outputs=n_outputs,
        )

    def build_constraint_from_relations(self, relations, predictor_names, output_names, col_offsets, n_features, base_mask=None):
        _ = relations, predictor_names, output_names, col_offsets, n_features, base_mask
        return None

    def build_regularizer(self, regularizer_spec):
        _ = regularizer_spec
        return None


class _CountingStationarity:
    def __init__(self):
        self._fit_transform_calls = 0

    @property
    def fit_transform_calls(self):
        return self._fit_transform_calls

    def fit_transform(self, data_list):
        self._fit_transform_calls += 1
        return [df.copy() for df in data_list]


class _IdentityCustomScaler(_BaseScaler):
    def fit_transform(self, data):
        self._fitted = True
        return np.asarray(data, dtype=np.float64)

    def transform(self, data):
        self._ensure_fitted()
        return np.asarray(data, dtype=np.float64)

    def inverse_transform(self, data):
        self._ensure_fitted()
        return np.asarray(data, dtype=np.float64)


def test_orchestrator_clones_callbacks_with_run_names_for_base_and_references():
    _DummyModel.seen_run_names = []

    old_get_strategy = orchestrator_module.BackendFactory.get_strategy
    orchestrator_module.BackendFactory.get_strategy = staticmethod(lambda backend: _DummyStrategy())

    try:
        api = MultiTaskGrangerAPI(backend="dummy")

        x = np.linspace(0.0, 1.0, 20)
        y = np.linspace(1.0, 0.0, 20)
        data = pd.DataFrame(np.column_stack([x, y]))
        data.columns = pd.Index(("x", "y"), dtype=object)

        api.fit(
            data=data,
            causes=["x", "y"],
            effects=["y"],
            tested_causes=["x", "y"],
            lag_config=LagConfiguration(max_lag=1, use_lag_zero=False),
            stationarity_transformer=StationarityTransformer(max_differencing_order=0),
            callbacks=[_DummyCallback()],
            model_config={"epochs": 1},
        )

        run_names = _DummyModel.seen_run_names
        assert "base_model" in run_names
        assert "reference_cause_x" in run_names
        assert "reference_cause_y" in run_names
    finally:
        orchestrator_module.BackendFactory.get_strategy = old_get_strategy


def test_orchestrator_uses_default_stationarity_and_reuses_prepared_data_explicitly():
    old_get_strategy = orchestrator_module.BackendFactory.get_strategy
    orchestrator_module.BackendFactory.get_strategy = staticmethod(lambda backend: _DummyStrategy())

    try:
        x = np.linspace(0.0, 1.0, 20)
        y = np.linspace(1.0, 0.0, 20)
        data = pd.DataFrame(np.column_stack([x, y]))
        data.columns = pd.Index(("x", "y"), dtype=object)

        api_default = MultiTaskGrangerAPI(backend="dummy")
        assert isinstance(api_default._stationarity_transformer, StationarityTransformer)

        counting_stationarity = _CountingStationarity()
        api = MultiTaskGrangerAPI(
            backend="dummy",
            stationarity_transformer=counting_stationarity,
            reuse_data=True,
        )

        common_kwargs = dict(
            data=data,
            causes=["x", "y"],
            effects=["y"],
            tested_causes=["x"],
            lag_config=LagConfiguration(max_lag=1, use_lag_zero=False),
            callbacks=[_DummyCallback()],
            model_config={"epochs": 1},
        )

        first_output = api.fit(**common_kwargs)
        second_output = api.fit(**common_kwargs, prepared_data=first_output.prepared_data)

        assert counting_stationarity.fit_transform_calls == 1
        assert first_output.prepared_data is not None
        assert second_output.prepared_data is not None
    finally:
        orchestrator_module.BackendFactory.get_strategy = old_get_strategy


def test_orchestrator_accepts_custom_scaler_class():
    old_get_strategy = orchestrator_module.BackendFactory.get_strategy
    orchestrator_module.BackendFactory.get_strategy = staticmethod(lambda backend: _DummyStrategy())

    try:
        api = MultiTaskGrangerAPI(backend="dummy")
        x = np.linspace(0.0, 1.0, 20)
        y = np.linspace(1.0, 0.0, 20)
        data = pd.DataFrame(np.column_stack([x, y]))
        data.columns = pd.Index(("x", "y"), dtype=object)

        prepared = api._prepare_data(
            data=data,
            effects=["y"],
            lag_config=LagConfiguration(max_lag=1, use_lag_zero=False),
            x_scaler=_IdentityCustomScaler,
            y_scaler=_IdentityCustomScaler,
        )

        assert isinstance(prepared.x_scaler, _IdentityCustomScaler)
        assert isinstance(prepared.y_scaler, _IdentityCustomScaler)
    finally:
        orchestrator_module.BackendFactory.get_strategy = old_get_strategy


if __name__ == "__main__":
    tests = [
        test_orchestrator_clones_callbacks_with_run_names_for_base_and_references,
        test_orchestrator_uses_default_stationarity_and_reuses_prepared_data_explicitly,
        test_orchestrator_accepts_custom_scaler_class,
    ]

    print("\n" + "=" * 80)
    print("ORCHESTRATOR CALLBACK TESTS")
    print("=" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except SkipTest as exc:
            print(f"SKIP: {name} -> {exc}")
            skipped += 1
        except Exception as exc:
            print(f"FAIL: {name} -> {exc}")
            traceback.print_exc(limit=2)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(
        f"Summary: {passed}/{total} passed, {failed}/{total} failed, {skipped}/{total} skipped"
    )
    print("=" * 80 + "\n")
