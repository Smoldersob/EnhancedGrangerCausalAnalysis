import traceback

import numpy as np
import pandas as pd

from ..core.exceptions import (
    ColumnMismatchError,
    DataShapeError,
    EmptyDataError,
    LagConfigurationError,
)
from ..utilities.validation import (
    validate_columns_present,
    validate_dataframe_list,
    validate_lag_bounds,
)


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


def test_validate_dataframe_list_accepts_matching_columns_and_shape():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

    validated, cols = validate_dataframe_list(
        [df1, df2],
        require_same_columns=True,
        require_same_shape=True,
    )

    assert cols == ["a", "b"]
    assert len(validated) == 2
    assert validated[0].shape == (3, 2)
    assert validated[1].shape == (3, 2)


def test_validate_dataframe_list_raises_on_column_mismatch():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})

    _assert_raises(
        ColumnMismatchError,
        validate_dataframe_list,
        [df1, df2],
        require_same_columns=True,
        require_same_shape=False,
        allow_superset_columns=False,
    )


def test_validate_dataframe_list_raises_on_shape_mismatch_when_required():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = pd.DataFrame({"a": [7, 8], "b": [9, 10]})

    _assert_raises(
        DataShapeError,
        validate_dataframe_list,
        [df1, df2],
        require_same_columns=True,
        require_same_shape=True,
    )


def test_validate_dataframe_list_allows_superset_and_reorders_columns():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"b": [30, 40], "a": [10, 20], "extra": [0, 0]})

    validated, cols = validate_dataframe_list(
        [df1, df2],
        require_same_columns=True,
        allow_superset_columns=True,
        copy=True,
    )

    assert cols == ["a", "b"]
    assert list(validated[1].columns) == ["a", "b"]
    assert validated[1].shape == (2, 2)


def test_validate_dataframe_list_raises_on_empty_list():
    _assert_raises(EmptyDataError, validate_dataframe_list, [])


def test_validate_columns_present_ok_and_fail():
    validate_columns_present(["x", "y", "z"], ["x", "z"], context="effects")

    _assert_raises(
        ColumnMismatchError,
        validate_columns_present,
        ["x", "y"],
        ["x", "z"],
        context="effects",
    )


def test_validate_lag_bounds_ok_and_fail():
    validate_lag_bounds(np.array([0, 1]), np.array([1, 2]))

    _assert_raises(
        LagConfigurationError,
        validate_lag_bounds,
        np.array([0, 1, 2]),
        np.array([1, 2]),
    )

    _assert_raises(
        LagConfigurationError,
        validate_lag_bounds,
        np.array([-1, 0]),
        np.array([1, 2]),
    )

    _assert_raises(
        LagConfigurationError,
        validate_lag_bounds,
        np.array([2, 1]),
        np.array([1, 2]),
    )


if __name__ == "__main__":
    tests = [
        test_validate_dataframe_list_accepts_matching_columns_and_shape,
        test_validate_dataframe_list_raises_on_column_mismatch,
        test_validate_dataframe_list_raises_on_shape_mismatch_when_required,
        test_validate_dataframe_list_allows_superset_and_reorders_columns,
        test_validate_dataframe_list_raises_on_empty_list,
        test_validate_columns_present_ok_and_fail,
        test_validate_lag_bounds_ok_and_fail,
    ]

    print("\n" + "=" * 80)
    print("VALIDATION TESTS")
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
