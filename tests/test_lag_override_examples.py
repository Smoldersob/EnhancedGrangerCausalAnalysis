import numpy as np
import pandas as pd

from ..core.lag_config import LagConfiguration
from ..preprocessing.lag.lag_engine import LagEngine
from ..preprocessing.lag.lag_selectors import BaseLagSelector


class FixedMatrixSelector(BaseLagSelector):
    """Selector returning a fixed per-pair lag matrix for deterministic tests."""

    def _select_lags(self, X, targets, ar_lags, pred_lag_matrix):
        matrix = np.array([[1, 2], [2, 1]], dtype=int)
        pred_lag_matrix[:, :] = matrix
        ar_lags[:] = np.array([1, 1], dtype=int)


def _print_case(case_name: str, engine: LagEngine, expected_mask: np.ndarray = None) -> None:
    """Pretty console dump to manually compare lag lists/masks."""
    widths = engine.lag_order_["max"] - engine.lag_order_["min"] + 1
    col_offsets = np.concatenate([[0], widths.cumsum(dtype=int)])[:-1]

    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)
    print("lag_order_.min:", engine.lag_order_["min"])
    print("lag_order_.max:", engine.lag_order_["max"])
    print("col_offsets:", col_offsets)

    if engine.selection_result_ is not None:
        print("selector max_lags_per_pred:", engine.selection_result_.max_lags_per_pred)
        print("selector mask:")
        print(engine.selection_result_.mask)

    print("final mask:")
    print(engine.mask_)

    if expected_mask is not None:
        print("expected mask:")
        print(expected_mask)
        print("mask == expected:", np.array_equal(engine.mask_, expected_mask))


def _run_engine(custom_lags=None, custom_pair_lags=None):
    cfg = LagConfiguration(
        max_lag=3,
        use_lag_zero=False,
        custom_lags=custom_lags or {},
        custom_pair_lags=custom_pair_lags or {},
    )
    selector = FixedMatrixSelector(max_lag=3, use_lag_zero=False)
    engine = LagEngine(config=cfg, selector=selector)

    # Always use the same variable order in all tests.
    # Construct DataFrame via column stack to avoid pandas StringArray edge-cases
    x1 = np.linspace(0.0, 1.0, 20)
    x2 = np.linspace(1.0, 0.0, 20)
    df = pd.DataFrame(np.column_stack([x1, x2]))
    df.columns = pd.Index(["x1", "x2"], dtype=object)
    engine.prepare([df], effects=["x1", "x2"])
    return engine


def test_custom_lag_x1_single_value_matches_example():
    engine = _run_engine(custom_lags={"x1": (3,)})
    expected = np.array(
        [
            [1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    _print_case("CASE 1: custom_lags={'x1': (3,)}", engine, expected)

    assert np.array_equal(engine.lag_order_["max"], np.array([3, 2]))
    assert np.array_equal(
        engine.mask_,
        expected,
    )


def test_custom_lag_x1_min_max_matches_example():
    engine = _run_engine(custom_lags={"x1": (2, 3)})
    expected = np.array(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )
    _print_case("CASE 2: custom_lags={'x1': (2, 3)}", engine, expected)

    assert np.array_equal(engine.lag_order_["max"], np.array([3, 2]))
    assert np.array_equal(engine.lag_order_["min"], np.array([2, 1]))

    col_offsets = np.concatenate(
        [[0], (engine.lag_order_["max"] - engine.lag_order_["min"] + 1).cumsum()]
    )
    assert np.array_equal(col_offsets[:-1], np.array([0, 2]))

    assert np.array_equal(
        engine.mask_,
        expected,
    )


def test_custom_pair_lag_x2_x1_single_value_matches_example():
    engine = _run_engine(custom_pair_lags={("x2", "x1"): (3,)})

    assert np.array_equal(engine.lag_order_["max"], np.array([3, 2]))

    col_offsets = np.concatenate(
        [[0], (engine.lag_order_["max"] - engine.lag_order_["min"] + 1).cumsum()]
    )
    assert np.array_equal(col_offsets[:-1], np.array([0, 3]))

    expected = np.array(
        [
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    _print_case("CASE 3: custom_pair_lags={('x2','x1'): (3,)}", engine, expected)
    assert np.array_equal(engine.mask_, expected)

    # New columns in x1 block are 0 for non-target row (x1 row),
    # and 1 for the selected pair row (x2 row).
    x1_start = int(col_offsets[0])
    baseline_x1_width = 2
    new_cols = slice(x1_start + baseline_x1_width, x1_start + 3)
    assert np.all(engine.mask_[0, new_cols] == 0)
    assert np.all(engine.mask_[1, new_cols] == 1)


def test_custom_pair_lag_x2_x1_min_max_matches_example():
    engine = _run_engine(custom_pair_lags={("x2", "x1"): (2, 3)})
    expected = np.array(
        [
            [1, 0, 0, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    _print_case("CASE 4: custom_pair_lags={('x2','x1'): (2, 3)}", engine, expected)

    assert np.array_equal(engine.lag_order_["max"], np.array([3, 2]))

    col_offsets = np.concatenate(
        [[0], (engine.lag_order_["max"] - engine.lag_order_["min"] + 1).cumsum()]
    )
    assert np.array_equal(col_offsets[:-1], np.array([0, 3]))

    assert np.array_equal(
        engine.mask_,
        expected,
    )


def test_selection_result_stays_selector_baseline_with_custom_overrides():
    engine = _run_engine(custom_lags={"x1": (3,)})
    _print_case("CASE 5: selection_result baseline vs final mask", engine)

    # Selector baseline should remain unchanged.
    assert np.array_equal(engine.selection_result_.max_lags_per_pred, np.array([2, 2]))
    assert np.array_equal(
        engine.selection_result_.mask,
        np.array(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 0],
            ],
            dtype=int,
        ),
    )

    # Final engine mask reflects overrides.
    assert not np.array_equal(engine.selection_result_.mask, engine.mask_)


if __name__ == "__main__":
    try:
        test_custom_lag_x1_single_value_matches_example()
        test_custom_lag_x1_min_max_matches_example()
        test_custom_pair_lag_x2_x1_single_value_matches_example()
        test_custom_pair_lag_x2_x1_min_max_matches_example()
        test_selection_result_stays_selector_baseline_with_custom_overrides()
        print("override example tests: OK")
    except AssertionError as e:
        print(f"override example tests: FAILED ({e})")
        raise
