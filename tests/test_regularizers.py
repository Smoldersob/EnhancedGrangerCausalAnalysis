import os
import sys
import traceback
from pathlib import Path
from importlib.util import find_spec

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.backends.regularizers.numpy_regularizers import (
    NumpyL1Regularizer,
    NumpyLagDependentL1Regularizer,
)


class SkipTest(Exception):
    """Local skip marker for optional runtime dependencies."""


def _require_torch():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")


def test_numpy_l1_regularizer_returns_penalty_and_gradient():
    reg = NumpyL1Regularizer(l1=0.5)
    weights = np.array([[1.0, -2.0], [0.0, 3.0]], dtype=np.float64)

    penalty, grad = reg(weights)

    assert np.isclose(penalty, 3.0)
    np.testing.assert_allclose(grad, 0.5 * np.sign(weights))


def test_numpy_lag_dependent_regularizer_weights_last_axis():
    reg = NumpyLagDependentL1Regularizer(
        l1=0.1,
        lag_weights=[0.0, 1.0, 2.0, 3.0],
        max_lags_per_pred=[3, 3],
        col_offsets=[0, 3],
    )
    weights = np.array([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]], dtype=np.float64)

    penalty, grad = reg(weights)

    expected_feature_weights = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    expected_penalty = 0.1 * np.sum(np.abs(weights) * expected_feature_weights)
    expected_grad = 0.1 * np.sign(weights) * expected_feature_weights

    assert np.isclose(penalty, expected_penalty)
    np.testing.assert_allclose(grad, expected_grad)


def test_numpy_lag_dependent_set_lag_layout_after_init():
    reg = NumpyLagDependentL1Regularizer(l1=1e-3)
    reg.set_lag_layout(max_lags_per_pred=[3, 3], col_offsets=[0, 3])

    params = reg.get_params()
    assert params["max_lags_per_pred"] == [3, 3]
    assert params["col_offsets"] == [0, 3]
    assert len(params["lag_weights"]) == 20


def test_numpy_lag_dependent_regularizer_shifted_lag_window_uses_absolute_lag_index():
    # Two blocks of length 3 with max_lag=4 represent lag window 2..4.
    reg = NumpyLagDependentL1Regularizer(
        l1=0.1,
        lag_weights=[100.0, 200.0, 1.0, 2.0, 3.0],
        max_lags_per_pred=[4, 4],
        col_offsets=[0, 3],
    )
    weights = np.array([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]], dtype=np.float64)

    penalty, grad = reg(weights)

    expected_feature_weights = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    expected_penalty = 0.1 * np.sum(np.abs(weights) * expected_feature_weights)
    expected_grad = 0.1 * np.sign(weights) * expected_feature_weights

    assert np.isclose(penalty, expected_penalty)
    np.testing.assert_allclose(grad, expected_grad)


def test_pytorch_regularizers_basic_behavior():
    _require_torch()

    import torch
    from complex_granger_analysis.backends.regularizers.pytorch_regularizers import (
        PyTorchL1Regularizer,
        PyTorchLagDependentL1Regularizer,
    )

    l1_reg = PyTorchL1Regularizer(l1=0.25)
    x = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    l1_penalty = l1_reg(x)
    assert torch.is_tensor(l1_penalty)
    assert np.isclose(float(l1_penalty.item()), 0.75)

    lag_reg = PyTorchLagDependentL1Regularizer(
        l1=0.1,
        lag_weights=[0.0, 1.0, 2.0, 3.0],
        max_lags_per_pred=[3, 3],
        col_offsets=[0, 3],
    )
    w = torch.tensor([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]], dtype=torch.float32)
    lag_penalty = lag_reg(w)
    expected = 0.1 * np.sum(np.abs(w.numpy()) * np.array([1, 2, 3, 1, 2, 3], dtype=np.float32))
    assert np.isclose(float(lag_penalty.item()), expected)

    lag_reg2 = PyTorchLagDependentL1Regularizer(l1=1e-3)
    lag_reg2.set_lag_layout(max_lags_per_pred=[5, 5], col_offsets=[0, 5])
    assert lag_reg2.get_params()["max_lags_per_pred"] == [5, 5]


def test_pytorch_lag_dependent_regularizer_shifted_lag_window_uses_absolute_lag_index():
    _require_torch()

    import torch
    from complex_granger_analysis.backends.regularizers.pytorch_regularizers import (
        PyTorchLagDependentL1Regularizer,
    )

    lag_reg = PyTorchLagDependentL1Regularizer(
        l1=0.1,
        lag_weights=[100.0, 200.0, 1.0, 2.0, 3.0],
        max_lags_per_pred=[4, 4],
        col_offsets=[0, 3],
    )
    w = torch.tensor([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0]], dtype=torch.float32)
    lag_penalty = lag_reg(w)

    expected_feature_weights = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float32)
    expected = 0.1 * np.sum(np.abs(w.numpy()) * expected_feature_weights)
    assert np.isclose(float(lag_penalty.item()), expected)


if __name__ == "__main__":
    tests = [
        test_numpy_l1_regularizer_returns_penalty_and_gradient,
        test_numpy_lag_dependent_regularizer_weights_last_axis,
        test_numpy_lag_dependent_set_lag_layout_after_init,
        test_numpy_lag_dependent_regularizer_shifted_lag_window_uses_absolute_lag_index,
        test_pytorch_regularizers_basic_behavior,
        test_pytorch_lag_dependent_regularizer_shifted_lag_window_uses_absolute_lag_index,
    ]

    print("\n" + "=" * 80)
    print("REGULARIZER TESTS")
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
            traceback.print_exc(limit=1)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed, {skipped}/{total} skipped")
    print("=" * 80 + "\n")
