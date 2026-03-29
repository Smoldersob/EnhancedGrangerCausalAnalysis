import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly from its nested location, e.g.:
# python complex_granger_analysis/tests/stationarity_mech_tests.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.preprocessing.stationarity.tests import (
    apply_differencing,
    static_adfuller_order,
    static_kpss_order,
)
from complex_granger_analysis.preprocessing.stationarity.transformer import (
    StationarityTransformer,
)


def _generate_ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=n)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_adf_order_for_low_autoregression_is_zero():
    """Low AR(1) should typically be already stationary, so order=0."""
    x = _generate_ar1(phi=0.2, n=1200, seed=42)
    s = pd.Series(x)

    order = static_adfuller_order(s, maxlag=5, alpha=0.05)

    assert order == 0


def test_kpss_order_for_low_autoregression_is_zero():
    """Low AR(1) should typically be already stationary, so order=0 (KPSS)."""
    x = _generate_ar1(phi=0.2, n=1200, seed=42)
    s = pd.Series(x)

    order = static_kpss_order(s, maxlag=5, alpha=0.05)

    assert order == 0


def test_apply_differencing_single_variable_known_values():
    """Validate differencing against a known, hand-computed sequence."""
    s = pd.Series([1.0, 4.0, 9.0, 16.0], name="x")

    d1 = apply_differencing(s, order=1)
    d2 = apply_differencing(s, order=2)

    expected_d1 = pd.Series([np.nan, 3.0, 5.0, 7.0], name="x")
    expected_d2 = pd.Series([np.nan, np.nan, 2.0, 2.0], name="x")

    pd.testing.assert_series_equal(d1, expected_d1)
    pd.testing.assert_series_equal(d2, expected_d2)


def test_fit_stationarity_uses_max_order_across_datasets():
    """Fitted order per variable should be max order seen over all datasets."""
    rng = np.random.default_rng(123)
    n = 350

    ds_stationary = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": _generate_ar1(phi=0.3, n=n, seed=7),
        }
    )
    ds_random_walk = pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(size=n)),
            "y": _generate_ar1(phi=0.3, n=n, seed=11),
        }
    )

    expected_x = max(
        static_adfuller_order(ds_stationary["x"], maxlag=5),
        static_adfuller_order(ds_random_walk["x"], maxlag=5),
    )
    expected_y = max(
        static_adfuller_order(ds_stationary["y"], maxlag=5),
        static_adfuller_order(ds_random_walk["y"], maxlag=5),
    )

    tr = StationarityTransformer(max_differencing_order=5, test_name="adf")
    tr.fit_stationarity([ds_stationary, ds_random_walk])

    assert tr.differencing_orders_["x"] == expected_x
    assert tr.differencing_orders_["y"] == expected_y


def test_fit_stationarity_kpss_uses_max_order_across_datasets():
    """KPSS variant: fitted order per variable should be max over datasets."""
    rng = np.random.default_rng(123)
    n = 350

    ds_stationary = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": _generate_ar1(phi=0.3, n=n, seed=7),
        }
    )
    ds_random_walk = pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(size=n)),
            "y": _generate_ar1(phi=0.3, n=n, seed=11),
        }
    )

    expected_x = max(
        static_kpss_order(ds_stationary["x"], maxlag=5),
        static_kpss_order(ds_random_walk["x"], maxlag=5),
    )
    expected_y = max(
        static_kpss_order(ds_stationary["y"], maxlag=5),
        static_kpss_order(ds_random_walk["y"], maxlag=5),
    )

    tr = StationarityTransformer(max_differencing_order=5, test_name="kpss")
    tr.fit_stationarity([ds_stationary, ds_random_walk])

    assert tr.differencing_orders_["x"] == expected_x
    assert tr.differencing_orders_["y"] == expected_y


def test_transform_applies_differencing_and_drops_nan_rows():
    """Transformed data should have expected leading NaN rows removed."""
    n = 120
    rng = np.random.default_rng(33)

    ds1 = pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(size=n)),
            "y": rng.normal(size=n),
        }
    )
    ds2 = pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(size=n)),
            "y": rng.normal(size=n),
        }
    )

    tr = StationarityTransformer(max_differencing_order=5, test_name="adf", dropna=True)
    out = tr.fit_transform([ds1, ds2])

    max_order = max(tr.differencing_orders_.values())

    assert len(out) == 2
    assert list(out[0].columns) == ["x", "y"]
    assert list(out[1].columns) == ["x", "y"]
    assert len(out[0]) == n - max_order
    assert len(out[1]) == n - max_order


def test_transform_kpss_applies_differencing_and_drops_nan_rows():
    """KPSS variant: transformed data should drop leading NaN rows."""
    n = 120
    rng = np.random.default_rng(33)

    ds1 = pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(size=n)),
            "y": rng.normal(size=n),
        }
    )
    ds2 = pd.DataFrame(
        {
            "x": np.cumsum(rng.normal(size=n)),
            "y": rng.normal(size=n),
        }
    )

    tr = StationarityTransformer(max_differencing_order=5, test_name="kpss", dropna=True)
    out = tr.fit_transform([ds1, ds2])

    max_order = max(tr.differencing_orders_.values())

    assert len(out) == 2
    assert list(out[0].columns) == ["x", "y"]
    assert list(out[1].columns) == ["x", "y"]
    assert len(out[0]) == n - max_order
    assert len(out[1]) == n - max_order


def test_prepare_static_backward_compatibility_shape_and_indices():
    """Legacy API should return nrows, cause indices and transformed datasets."""
    rng = np.random.default_rng(99)
    n = 180
    cols = ["a", "b", "c"]

    ds = pd.DataFrame(
        {
            "a": np.cumsum(rng.normal(size=n)),
            "b": rng.normal(size=n),
            "c": _generate_ar1(phi=0.25, n=n, seed=5),
        }
    )

    tr = StationarityTransformer(max_differencing_order=5, test_name="adf")
    nrows, columns_id, transformed = tr.prepare_static(
        [ds],
        causes=["b", "c"],
        effects=["a", "c"],
    )

    assert nrows == 2
    assert columns_id == [1, 2]
    assert len(transformed) == 1
    assert list(transformed[0].columns) == cols
    assert len(transformed[0]) <= n


if __name__ == "__main__":
    tests = [
        test_adf_order_for_low_autoregression_is_zero,
        test_kpss_order_for_low_autoregression_is_zero,
        test_apply_differencing_single_variable_known_values,
        test_fit_stationarity_uses_max_order_across_datasets,
        test_fit_stationarity_kpss_uses_max_order_across_datasets,
        test_transform_applies_differencing_and_drops_nan_rows,
        test_transform_kpss_applies_differencing_and_drops_nan_rows,
        test_prepare_static_backward_compatibility_shape_and_indices,
    ]

    print("\n" + "=" * 80)
    print("STATIONARITY MECHANISM TESTS")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as exc:  # keep running and show full summary at end
            print(f"FAIL: {name} -> {exc}")
            traceback.print_exc(limit=1)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 80 + "\n")
