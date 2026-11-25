from typing import Any

import numpy as np
import pandas as pd
import pytest

from enhanced_granger_analysis.preprocessing.lag.lag_selectors import (
    CommonOrderICLagSelector,
    ICLagSelector,
    NaiveVARLagSelector,
)
from enhanced_granger_analysis.preprocessing.lag.lag_selectors._space import (
    LagSelectionCapabilityError,
)
from enhanced_granger_analysis.preprocessing.lag.space import (
    LagSelectionSpace,
)


def _space(outputs, predictors, allowed, required=None, fixed=None):
    required = required or {}
    fixed = fixed or {}

    def complete(values):
        return {
            output: {
                predictor: tuple(values.get(output, {}).get(predictor, ()))
                for predictor in predictors
            }
            for output in outputs
        }

    allowed_lags = complete(allowed)
    required_lags = complete(required)
    for output in outputs:
        for predictor in predictors:
            allowed_lags[output][predictor] = tuple(sorted(set(
                allowed_lags[output][predictor]
                + required_lags[output][predictor]
                + tuple(fixed.get(output, {}).get(predictor, ()))
            )))

    return LagSelectionSpace.resolve(
        outputs=outputs,
        predictors=predictors,
        max_lag=8,
        allowed_lags=allowed_lags,
        required_lags=required_lags,
        fixed_lags=fixed,
    )


def _data(n=80):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "response_only": rng.normal(size=n),
        "z": rng.normal(size=n),
    })


def test_incremental_scored_candidates_obey_sparse_plan_and_predictors():
    selector = ICLagSelector(
        max_lag=3,
        min_ar_order=0,
        prune_lags=False,
    )
    scored = []
    original_evaluate = selector._evaluate

    def capture(y, lags, columns, *, target, stage):
        scored.append({name: tuple(values) for name, values in lags.items()})
        return original_evaluate(
            y, lags, columns, target=target, stage=stage
        )

    selector._evaluate = capture
    plan = _space(
        ("y",),
        ("x", "y"),
        {"y": {"x": (1, 3), "y": (1,)}},
    )

    result = selector.fit(_data(), outputs=("y",), space=plan)

    assert result["y"]["x"] in {(), (1,), (1, 3)}
    assert scored
    assert all(2 not in model["x"] for model in scored)
    assert all("response_only" not in model for model in scored)


def test_required_lag_survives_incremental_pruning():
    class PruneOptionalTerms(ICLagSelector):
        def _score(self, y, design):
            return -float(design.shape[1])

        def _accept_addition(self, candidate_score, reference_score):
            return True

        def _accept_removal(self, candidate_score, reference_score):
            return True

    selector = PruneOptionalTerms(
        max_lag=3,
        min_ar_order=0,
        prune_lags=True,
    )
    plan = _space(
        ("y",),
        ("x",),
        {"y": {"x": (1, 2, 3)}},
        required={"y": {"x": (1,)}},
    )

    result = selector.fit(_data(), outputs=("y",), space=plan)

    assert result["y"]["x"] == (1,)


def test_common_order_scores_sparse_lags_and_exact_fixed_relation():
    selector = CommonOrderICLagSelector(max_lag=3)
    designs = []
    original_score_order = selector._score_order

    def capture(responses, candidate_designs, weights):
        designs.extend(candidate_designs)
        return original_score_order(responses, candidate_designs, weights)

    selector._score_order = capture
    plan = _space(
        ("y",),
        ("x", "z"),
        {"y": {"x": (1, 3), "z": (1, 3)}},
        fixed={"y": {"x": (5,)}},
    )

    data = _data()
    result = selector.fit(data, outputs=("y",), space=plan)

    assert designs
    assert all(design.shape[1] in {1, 2, 3} for design in designs)
    expected_fixed_column = data["x"].to_numpy()[:-5]
    assert all(
        np.array_equal(design[:, 0], expected_fixed_column)
        for design in designs
    )
    assert result["y"]["x"] == (5,)
    assert 2 not in result["y"]["z"]


def test_target_only_response_is_not_added_as_a_predictor():
    selector = ICLagSelector(
        max_lag=2,
        min_ar_order=0,
        prune_lags=False,
    )
    evaluated = []
    original_evaluate = selector._evaluate

    def capture(y, lags, columns, *, target, stage):
        evaluated.append(dict(lags))
        return original_evaluate(
            y, lags, columns, target=target, stage=stage
        )

    selector._evaluate = capture
    plan = _space(
        ("response_only",),
        ("x",),
        {"response_only": {"x": (1,)}},
    )

    result = selector.fit(_data(), outputs=("response_only",), space=plan)

    assert tuple(result) == ("response_only",)
    assert tuple(result["response_only"]) == ("x",)
    assert evaluated
    assert all(tuple(model) == ("x",) for model in evaluated)


def test_incremental_fixed_relation_is_context_not_a_selectable_pair():
    selector = ICLagSelector(
        max_lag=3,
        min_ar_order=0,
        prune_lags=False,
    )
    evaluated = []
    original_evaluate = selector._evaluate

    def capture(y, lags, columns, *, target, stage):
        evaluated.append({name: tuple(values) for name, values in lags.items()})
        return original_evaluate(
            y, lags, columns, target=target, stage=stage
        )

    selector._evaluate = capture
    plan = _space(
        ("y",),
        ("x", "z"),
        {"y": {"x": (1, 3), "z": (1, 3)}},
        fixed={"y": {"x": (5,)}},
    )

    result = selector.fit(_data(), outputs=("y",), space=plan)

    assert result["y"]["x"] == (5,)
    assert evaluated
    assert all(model["x"] == (5,) for model in evaluated)
    assert any(model["z"] for model in evaluated)
    assert all(2 not in model["z"] for model in evaluated)


def test_empty_allowed_or_fixed_relation_is_not_widened():
    selector = ICLagSelector(
        max_lag=3,
        min_ar_order=0,
        prune_lags=False,
    )
    empty_allowed = _space(
        ("y",),
        ("x",),
        {"y": {"x": ()}},
    )
    assert selector.fit(
        _data(), outputs=("y",), space=empty_allowed
    )["y"]["x"] == ()

    fixed_empty = _space(
        ("y",),
        ("x",),
        {"y": {"x": (1, 3)}},
        fixed={"y": {"x": ()}},
    )
    assert selector.fit(
        _data(), outputs=("y",), space=fixed_empty
    )["y"]["x"] == ()


def test_noncanonical_plan_is_rejected_before_scoring():
    class UnsupportedPlan:
        outputs = ("y",)
        predictors = ("x",)
        allowed_lags = {"y": {"x": (1,)}}
        required_lags = {"y": {"x": ()}}
        fixed_lags = {}

    selector = ICLagSelector(max_lag=2, min_ar_order=0)
    unsupported: Any = UnsupportedPlan()

    with pytest.raises(TypeError, match="LagSelectionSpace"):
        selector.fit(_data(), outputs=("y",), space=unsupported)


def test_naive_var_rejects_asymmetric_plan_before_fitting():
    selector = NaiveVARLagSelector(max_lag=2)
    plan = _space(
        ("x", "y"),
        ("x", "y"),
        {
            "x": {"x": (1, 2), "y": (1,)},
            "y": {"x": (1, 2), "y": (1, 2)},
        },
    )

    with pytest.raises(
        LagSelectionCapabilityError,
        match="asymmetric or sparse plans",
    ):
        selector.fit(_data(), outputs=("x", "y"), space=plan)
