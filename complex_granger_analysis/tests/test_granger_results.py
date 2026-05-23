import traceback
from importlib.util import find_spec

import numpy as np

if find_spec("pandas") is not None:
    from ..results.granger_results import GrangerAnalysisResults
else:
    GrangerAnalysisResults = None


class SkipTest(Exception):
    pass


class DummyModel:
    """Minimal model stub exposing predict() and get_weights()."""

    def __init__(self, predictions: np.ndarray, kernel_f_by_o: np.ndarray) -> None:
        self._predictions = np.asarray(predictions, dtype=np.float64)
        self._kernel = np.asarray(kernel_f_by_o, dtype=np.float64)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != self._predictions.shape[0]:
            raise ValueError("Prediction rows mismatch")
        return self._predictions

    def get_weights(self):
        return [self._kernel]


def test_granger_results_updates_sign_and_p_value_with_prediction_inputs():
    if GrangerAnalysisResults is None:
        raise SkipTest("pandas is not installed")

    effects = ["y1", "y2"]
    causes = ["x1", "x2"]
    result = GrangerAnalysisResults(effects=effects, causes=causes)

    n_samples = 10
    y_true = np.column_stack([
        np.linspace(0.0, 1.0, n_samples),
        np.linspace(1.0, 0.0, n_samples),
    ])

    # Base predictions close to target (lower error), ref predictions worse.
    y_base = y_true + 0.01
    y_ref = y_true + 0.50

    # Kernel shape: (n_features, n_outputs)
    # Cause x1 block -> columns [0,1,2]
    # Output y1 block weights: [0.2, -0.9, 0.4] => sign should be -1
    # Output y2 block weights: [0.1, 0.3, -0.8] => sign should be -1
    base_kernel = np.array(
        [
            [0.2, 0.1],
            [-0.9, 0.3],
            [0.4, -0.8],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ref_kernel = np.zeros_like(base_kernel)

    base_model = DummyModel(predictions=y_base, kernel_f_by_o=base_kernel)
    ref_model = DummyModel(predictions=y_ref, kernel_f_by_o=ref_kernel)

    col_offsets = np.array([0, 3, 6], dtype=int)
    result.update_cause(
        cause="x1",
        cause_index=0,
        col_offsets=col_offsets,
        y_true=y_true,
        base_predictions=y_base,
        reference_predictions=y_ref,
        base_weights=base_kernel,
        reference_weights=ref_kernel,
    )

    sign_col = result.sign.loc[:, "x1"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(sign_col, np.array([-1.0, -1.0], dtype=np.float64))

    p_values = result.p_value.loc[:, "x1"].to_numpy(dtype=np.float64)
    assert np.all(p_values >= 0.0)
    assert np.all(p_values <= 1.0)


def test_granger_results_stores_base_and_reference_weights_and_predictions():
    if GrangerAnalysisResults is None:
        raise SkipTest("pandas is not installed")

    result = GrangerAnalysisResults(effects=["y1"], causes=["x1"])

    y = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y_base = y + 0.01
    y_ref = y + 0.10

    base_kernel = np.array([[0.5], [0.2], [-0.1]], dtype=np.float64)
    ref_kernel = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    base_model = DummyModel(predictions=y_base, kernel_f_by_o=base_kernel)
    ref_model = DummyModel(predictions=y_ref, kernel_f_by_o=ref_kernel)

    result.update_cause(
        cause="x1",
        cause_index=0,
        col_offsets=np.array([0, 3], dtype=int),
        y_true=y,
        base_predictions=y_base,
        reference_predictions=y_ref,
        base_weights=base_kernel,
        reference_weights=ref_kernel,
    )

    assert result.base_weights is not None
    assert result.base_predictions is not None
    assert "x1" in result.ref_weights
    assert "x1" in result.ref_predictions
    assert result.ref_predictions["x1"].shape == y.shape


if __name__ == "__main__":
    tests = [
        test_granger_results_updates_sign_and_p_value_with_prediction_inputs,
        test_granger_results_stores_base_and_reference_weights_and_predictions,
    ]

    print("\n" + "=" * 80)
    print("GRANGER RESULTS TESTS")
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
