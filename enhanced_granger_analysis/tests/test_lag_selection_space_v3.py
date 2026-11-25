"""Focused tests for resolved lag spaces and LagEngine integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..preprocessing.configuration.lag import LagEngineConfig
from ..preprocessing.lag.lag_engine import LagEngine
from ..preprocessing.lag.registry import UnsupportedLagSelectionSpaceError
from ..preprocessing.lag.space import LagSelectionSpace


def _frame(offset: float = 0.0) -> pd.DataFrame:
    values = np.arange(20, dtype=float) + offset
    return pd.DataFrame({"a": values, "y": values * 2})


def _space(**kwargs) -> LagSelectionSpace:
    return LagSelectionSpace.resolve(
        outputs=("y",),
        predictors=("a",),
        max_lag=3,
        **kwargs,
    )


class _PlanSelector:
    selection_space_capabilities = {
        "allowed_lags",
        "required_lags",
        "fixed_lags",
    }

    seen_spaces = []

    def __init__(self, result=None):
        self.result = result

    def fit(self, data, *, outputs=None, space=None):
        self.seen_spaces.append(space)
        if self.result is not None:
            return self.result
        return {
            output: {
                predictor: (
                    (1, 3) if predictor == "a" else ()
                )
                for predictor in data.columns
            }
            for output in outputs
        }


class _LegacySelector:
    fit_calls = []

    def fit(self, data, *, outputs=None):
        self.fit_calls.append(True)
        return {
            output: {
                predictor: (1,) if predictor != output else (1,)
                for predictor in data.columns
            }
            for output in outputs
        }


class _ExternalKeywordOnlySelector(_PlanSelector):
    seen_spaces = []

    def fit(self, data, *, outputs=None, selection_space=None):
        self.seen_spaces.append(selection_space)
        return super().fit(data, outputs=outputs)


class _DatasetDependentSelector(_PlanSelector):
    seen_spaces = []

    def fit(self, data, *, outputs=None, space=None):
        self.seen_spaces.append(space)
        lag = 1 if data["a"].iloc[0] < 50 else 3
        return {
            output: {
                predictor: (lag,) if predictor == "a" else ()
                for predictor in data.columns
            }
            for output in outputs
        }


def test_empty_allowed_pair_stays_empty_in_final_layout():
    space = _space(allowed_lags={"y": {"a": ()}})
    fitted = LagEngine(
        LagEngineConfig(max_lag=3),
    ).fit([_frame()], outputs=("y",), predictors=("a",), selection_space=space)

    assert fitted.final_lags == {"y": {"a": ()}}
    assert fitted.feature_predictors == ()
    assert fitted.mask.shape == (1, 0)


def test_fixed_empty_pair_and_zero_lag_are_preserved_in_feature_layout():
    empty_space = _space(
        fixed_lags={"y": {"a": ()}},
    )
    empty_fitted = LagEngine(LagEngineConfig(max_lag=3)).fit(
        [_frame()],
        outputs=("y",),
        predictors=("a",),
        selection_space=empty_space,
    )
    assert empty_fitted.final_lags["y"]["a"] == ()
    assert empty_fitted.mask.shape == (1, 0)

    frame = pd.DataFrame(
        {
            "a": np.arange(8, dtype=float),
            "b": np.arange(8, dtype=float) + 10,
            "y": np.arange(8, dtype=float) + 20,
            "z": np.arange(8, dtype=float) + 30,
        }
    )
    space = LagSelectionSpace.resolve(
        outputs=("y", "z"),
        predictors=("a", "b"),
        max_lag=2,
        use_lag_zero=True,
        allowed_lags={
            "y": {"a": (0, 2), "b": ()},
            "z": {"a": (2,), "b": (1,)},
        },
    )
    engine = LagEngine(
        LagEngineConfig(max_lag=2, use_lag_zero=True),
    )
    fitted = engine.fit(
        [frame],
        outputs=("y", "z"),
        predictors=("a", "b"),
        selection_space=space,
    )
    lagged = engine.transform([frame])

    assert tuple(zip(fitted.feature_predictors, fitted.feature_lags)) == (
        ("a", 0),
        ("a", 2),
        ("b", 1),
    )
    np.testing.assert_array_equal(
        fitted.mask,
        [[True, True, False], [False, True, True]],
    )
    np.testing.assert_array_equal(
        lagged.X,
        np.column_stack((frame["a"].to_numpy()[2:], frame["a"].to_numpy()[:-2], frame["b"].to_numpy()[1:-1])),
    )


def test_sparse_allowed_lags_are_not_expanded():
    space = _space(allowed_lags={"y": {"a": (1, 3)}})
    fitted = LagEngine(LagEngineConfig(max_lag=3)).fit(
        [_frame()], outputs=("y",), predictors=("a",), selection_space=space
    )

    assert fitted.final_lags["y"]["a"] == (1, 3)
    np.testing.assert_array_equal(fitted.feature_lags, [1, 3])


def test_required_lag_is_validated_and_fixed_relation_is_exact():
    space = _space(
        required_lags={"y": {"a": (2,)}},
        fixed_lags={"y": {"a": (2,)}},
    )
    fitted = LagEngine(LagEngineConfig(max_lag=3)).fit(
        [_frame()], outputs=("y",), predictors=("a",), selection_space=space
    )

    assert fitted.final_lags["y"]["a"] == (2,)
    assert fitted.selection_space is space

    invalid = _PlanSelector(
        {"y": {"a": (1, 2), "y": ()}}
    )
    with pytest.raises(ValueError, match="fixed relation"):
        LagEngine(LagEngineConfig(max_lag=3), invalid).fit(
            [_frame()],
            outputs=("y",),
            predictors=("a",),
            selection_space=space,
        )


def test_overrides_are_post_selection_corrections_within_the_space():
    space = _space(allowed_lags={"y": {"a": (1, 3)}})
    selector = _PlanSelector()
    fitted = LagEngine(
        LagEngineConfig(
            max_lag=3,
            pair_overrides={"y": {"a": (3,)}},
        ),
        selector,
    ).fit(
        [_frame()], outputs=("y",), predictors=("a",), selection_space=space
    )

    assert fitted.aggregated_lags["y"]["a"] == (1, 3)
    assert fitted.final_lags["y"]["a"] == (3,)
    assert fitted.overridden_pairs == (("y", "a"),)

    invalid_engine = LagEngine(
        LagEngineConfig(
            max_lag=3,
            pair_overrides={"y": {"a": (2,)}},
        ),
        _PlanSelector(),
    )
    with pytest.raises(ValueError, match="forbidden lags"):
        invalid_engine.fit(
            [_frame()],
            outputs=("y",),
            predictors=("a",),
            selection_space=space,
        )


def test_selector_capability_is_checked_before_fit():
    _LegacySelector.fit_calls.clear()
    selector = _LegacySelector()
    with pytest.raises(UnsupportedLagSelectionSpaceError, match="No unrestricted fallback"):
        LagEngine(LagEngineConfig(max_lag=3), selector).fit(
            [_frame()],
            outputs=("y",),
            predictors=("a",),
            selection_space=_space(),
        )
    assert selector.fit_calls == []

    keyword_mismatch = _ExternalKeywordOnlySelector()
    keyword_mismatch.seen_spaces.clear()
    with pytest.raises(
        UnsupportedLagSelectionSpaceError,
        match="space fit argument",
    ):
        LagEngine(LagEngineConfig(max_lag=3), keyword_mismatch).fit(
            [_frame()],
            outputs=("y",),
            predictors=("a",),
            selection_space=_space(),
        )
    assert keyword_mismatch.seen_spaces == []


def test_selector_output_must_honor_space_and_response_only_pairs_are_closed():
    selector = _PlanSelector(
        {"y": {"a": (2,), "y": ()}}
    )
    space = _space(allowed_lags={"y": {"a": (1, 3)}})
    with pytest.raises(ValueError, match="forbidden lags"):
        LagEngine(LagEngineConfig(max_lag=3), selector).fit(
            [_frame()],
            outputs=("y",),
            predictors=("a",),
            selection_space=space,
        )

    selector = _PlanSelector()
    selector.seen_spaces.clear()
    fitted = LagEngine(LagEngineConfig(max_lag=3), selector).fit(
        [_frame()],
        outputs=("y",),
        predictors=("a",),
        selection_space=space,
    )
    assert selector.seen_spaces[-1].allowed_lags["y"]["y"] == ()
    assert fitted.final_lags["y"] == {"a": (1, 3)}


def test_selector_aggregation_remains_separate_from_space_and_overrides():
    space = _space(allowed_lags={"y": {"a": (1, 3)}})
    fitted = LagEngine(
        LagEngineConfig(
            max_lag=3,
            aggregation="union",
            pair_overrides={"y": {"a": (3,)}},
        ),
        _DatasetDependentSelector(),
    ).fit(
        [_frame(), _frame(offset=100)],
        outputs=("y",),
        predictors=("a",),
        selection_space=space,
    )

    assert fitted.selection_results == (
        {"y": {"a": (1,)}},
        {"y": {"a": (3,)}},
    )
    assert fitted.aggregated_lags == {"y": {"a": (1, 3)}}
    assert fitted.final_lags == {"y": {"a": (3,)}}


def test_resolved_space_rejects_conflicting_contracts_and_wrong_axes():
    with pytest.raises(ValueError, match="must be in allowed_lags"):
        _space(
            allowed_lags={"y": {"a": (1,)}},
            required_lags={"y": {"a": (3,)}},
        )

    wrong_axes = LagSelectionSpace.resolve(
        outputs=("y",), predictors=("other",), max_lag=3
    )
    with pytest.raises(ValueError, match="axes must exactly match"):
        LagEngine(LagEngineConfig(max_lag=3)).fit(
            [_frame()],
            outputs=("y",),
            predictors=("a",),
            selection_space=wrong_axes,
        )


def test_no_space_keeps_legacy_selector_and_baseline_paths():
    _LegacySelector.fit_calls.clear()
    fitted = LagEngine(
        LagEngineConfig(max_lag=2, use_lag_zero=True)
    ).fit([_frame()], outputs=("y",), predictors=("a",))
    assert fitted.final_lags["y"]["a"] == (0, 1, 2)

    selector = _LegacySelector()
    selected = LagEngine(LagEngineConfig(max_lag=3), selector).fit(
        [_frame()], outputs=("y",), predictors=("a",)
    )
    assert selected.final_lags["y"]["a"] == (1,)
    assert _LegacySelector.fit_calls == [True]
