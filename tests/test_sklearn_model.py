import sys
import traceback
from pathlib import Path

import numpy as np

# Allow running this file directly from its nested location, e.g.:
# python complex_granger_analysis/tests/test_sklearn_model.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.core.exceptions import TrainingError
from complex_granger_analysis.components.models.scikit_model import SklearnGrangerModel


def _assert_raises(exc_type, fn, *args, **kwargs):
    """Minimal helper to assert exception without pytest dependency in test body."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {exc_type.__name__}, but got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def test_sklearn_model_initialize_fit_and_omit_variables():
    rng = np.random.default_rng(999)
    X = rng.normal(size=(50, 8)).astype(np.float64)
    W = rng.normal(size=(8, 2)).astype(np.float64)
    y = X @ W

    model = SklearnGrangerModel(fit_intercept=True)
    model.initialize(X, lags=3, targets=y)

    # Check initial mask is all ones
    assert np.allclose(model._variable_mask, 1.0)  # pylint: disable=protected-access

    # Omit variables 2 and 5
    model.omit_variables([2, 5])
    mask = model._variable_mask  # pylint: disable=protected-access
    assert mask[2] == 0.0
    assert mask[5] == 0.0
    assert mask[0] == 1.0

    result = model.fit()
    assert "weights" in result
    assert "forecasts" in result
    assert "history" in result
    assert result["forecasts"].shape == y.shape
    assert len(result["weights"]) == 1
    assert isinstance(result["test_statistic"], float)


def test_sklearn_model_initialize_requires_targets():
    X = np.random.randn(15, 6)
    model = SklearnGrangerModel()

    _assert_raises(TrainingError, model.initialize, X, 2)


def test_sklearn_model_hyperoptimize_returns_message():
    model = SklearnGrangerModel()
    result = model.hyperoptimize({"alpha": [0.1, 1.0]}, n_trials=5)

    assert isinstance(result, dict)
    assert "message" in result
    assert "nie posiada parametrów do hiperoptymalizacji" in result["message"]


def test_sklearn_model_set_and_get_weights():
    rng = np.random.default_rng(111)
    X = rng.normal(size=(30, 4)).astype(np.float64)
    y = rng.normal(size=(30, 1)).astype(np.float64)

    model = SklearnGrangerModel()
    model.initialize(X, lags=1, targets=y)
    model.fit()

    weights = model.get_weights()
    assert len(weights) == 1
    assert weights[0].shape[0] == 4  # n_features
    assert weights[0].shape[1] == 1  # n_outputs

    # Set new weights
    new_weights = np.random.randn(4, 1).astype(np.float64)
    model.set_weights(new_weights)
    retrieved = model.get_weights()
    assert np.allclose(retrieved[0], new_weights)


if __name__ == "__main__":
    tests = [
        test_sklearn_model_initialize_fit_and_omit_variables,
        test_sklearn_model_initialize_requires_targets,
        test_sklearn_model_hyperoptimize_returns_message,
        test_sklearn_model_set_and_get_weights,
    ]

    print("\n" + "=" * 80)
    print("SKLEARN MODEL TESTS")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {name} -> {exc}")
            traceback.print_exc(limit=1)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 80 + "\n")
