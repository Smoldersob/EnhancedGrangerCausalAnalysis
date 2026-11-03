"""Regression tests for the paired execution workflow."""
from types import SimpleNamespace

import numpy as np
import pytest

from enhanced_granger_analysis.execution.runner import AnalysisRunner, AnalysisRunProgress
from enhanced_granger_analysis.execution.granger_trainer import GrangerTrainer
from enhanced_granger_analysis.results.granger_results import GrangerAnalysisResults
from enhanced_granger_analysis.core.exceptions import DataValidationError


class DummyModel:
    def __init__(self, role):
        self.base_model_role = role
        self.use_bias = True
        self.input_spec = SimpleNamespace(n_outputs=1, n_features=2)
        self.weights = [np.zeros((2, 1)), np.zeros(1)]
        self.is_fitted = False
        self.omitted = []

    def get_weights(self):
        return [w.copy() for w in self.weights]

    def set_weights(self, weights):
        self.weights = [w.copy() for w in weights]
        self.is_fitted = False

    def add_variables(self, indices):
        self.omitted = []

    def omit_variables(self, indices):
        self.omitted = list(indices)

    def get_active_mask(self):
        mask = np.ones((1, 2), dtype=bool)
        mask[:, self.omitted] = False
        return mask

    def predict(self, X):
        if not self.is_fitted:
            raise AssertionError("Prediction attempted after fitted state was invalidated")
        return X @ (self.weights[0] * self.get_active_mask().T) + self.weights[1]


class DummyTrainer(GrangerTrainer):
    def __init__(self):
        self.calls = []
        self.starts = []
        self.fit_number = 0

    def _fit(self, model, prepared, cause, role, base_weights, initial_weights):
        source = base_weights if base_weights is not None else initial_weights
        if source is not None:
            model.set_weights(source)
        self.starts.append((role, cause, model.get_weights()))
        self.calls.append((role, cause))
        self.fit_number += 1
        model.add_variables([]) if role != "restricted" else model.omit_variables(
            [0] if cause == "a" else [1])
        model.weights = [np.full((2, 1), float(self.fit_number)), np.array([float(self.fit_number)])]
        model.is_fitted = True
        return self._capture_outcome(model=model, X=prepared.X_scaled,
            training_result=SimpleNamespace(), timings={})

    def fit_base(self, *, model, prepared, initial_weights=None):
        return self._fit(model, prepared, None, model.base_model_role, None, initial_weights)

    def fit_unrestricted(self, *, model, prepared, cause, base_weights=None, initial_weights=None):
        return self._fit(model, prepared, cause, "unrestricted", base_weights, initial_weights)

    def fit_restricted(self, *, model, prepared, cause, base_weights=None, initial_weights=None):
        return self._fit(model, prepared, cause, "restricted", base_weights, initial_weights)


def setup_case(role):
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    prepared = SimpleNamespace(
        X_scaled=X, X_raw=X, tested_causes=["a", "b"], causes=["a", "b"],
        cause_slices={"a": slice(0, 1), "b": slice(1, 2)},
        y_scaler=SimpleNamespace(inverse_transform=lambda y: y.copy()),
        indices_for=lambda *args, **kwargs: np.arange(len(X)),
    )
    results = GrangerAnalysisResults(effects=["y"], causes=["a", "b"],
        mask=np.ones((1, 2), dtype=bool), cause_slices=prepared.cause_slices,
        X_true=X, y_true=np.zeros((3, 1)))
    trainer = DummyTrainer()
    runner = AnalysisRunner(trainer=trainer)
    return runner, trainer, DummyModel(role), prepared, AnalysisRunProgress(results=results)


@pytest.mark.parametrize("role", [None, "unrestricted", "restricted"])
@pytest.mark.parametrize("only_u", [False, True])
def test_training_order_and_snapshot_provenance(role, only_u):
    runner, trainer, model, prepared, progress = setup_case(role)
    runner.run(model=model, prepared=prepared, progress=progress,
               train_only_unrestricted=only_u)
    if role == "unrestricted":
        assert (
            progress.results.get_snapshot("a", "unrestricted")
            is progress.results.get_snapshot("b", "unrestricted")
        )

    elif role == "restricted" and not only_u:
        assert (
            progress.results.get_snapshot("a", "restricted")
            is progress.results.get_snapshot("b", "restricted")
        )

    elif role == "restricted" and only_u:
        assert progress.base_weights is None

        for snapshot_role in ("unrestricted", "restricted"):
            assert (
                progress.results.get_snapshot("a", snapshot_role)
                is not progress.results.get_snapshot("b", snapshot_role)
            )
            
    for cause in prepared.tested_causes:
        u = progress.results.get_snapshot(cause, "unrestricted")
        r = progress.results.get_snapshot(cause, "restricted")
        assert r.metadata["fit_performed"] is (not only_u)
        if only_u:
            np.testing.assert_array_equal(r.weights, u.weights)
            expected_prediction = prepared.X_scaled @ np.where(r.active_slope_mask.T,
                u.weights.T, 0.) + u.weights[0, 0]
            np.testing.assert_array_equal(r.predictions, expected_prediction)
    if role == "unrestricted":
        assert progress.results.get_snapshot("a", "unrestricted") is progress.results.get_snapshot("b", "unrestricted")
    elif role == "restricted" and only_u:
        assert progress.base_weights is None

        assert (
            progress.results.get_snapshot("a", "unrestricted")
            is not progress.results.get_snapshot("b", "unrestricted")
        )

        assert (
            progress.results.get_snapshot("a", "restricted")
            is not progress.results.get_snapshot("b", "restricted")
        )


def test_no_baseline_initialization_and_pair_warm_start():
    runner, trainer, model, prepared, progress = setup_case(None)
    runner.run(model=model, prepared=prepared, progress=progress)
    for role, cause, weights in trainer.starts:
        if role == "unrestricted":
            np.testing.assert_array_equal(weights[0], np.zeros((2, 1)))
        else:
            np.testing.assert_array_equal(weights[0].T,
                progress.results.get_snapshot(cause, "unrestricted").weights)


def test_capture_rejects_unmatched_source_weights():
    runner, trainer, model, prepared, progress = setup_case(None)
    outcome = trainer.fit_unrestricted(model=model, prepared=prepared, cause="a")
    wrong = [w + 1 for w in outcome.weights]
    with pytest.raises(DataValidationError):
        trainer.capture_restricted(model=model, prepared=prepared, cause="a", source_weights=wrong)
