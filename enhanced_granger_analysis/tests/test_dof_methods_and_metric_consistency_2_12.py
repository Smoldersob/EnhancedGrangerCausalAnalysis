"""Tests for registered DOF strategies and metric consistency.

The tests build small exact-OLS comparisons so that the classical
identities between F, coefficient Wald, projection Wald, and LR hold
numerically:

    W_coefficient = q * F,   W_projection = q * F,
    LR = n * log(1 + q * F / df).

A separate collinear fixture verifies that rank-based DOF detects a
redundant removed column where structural counting does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from enhanced_granger_analysis.core.exceptions import ResultsError
from enhanced_granger_analysis.results.dof import (
    DofResult,
    available_dof,
    register_dof,
    resolve_dof,
)
from enhanced_granger_analysis.results.granger_results import (
    GrangerAnalysisResults,
    ModelSnapshot,
)


N_SAMPLES = 60
FEATURES = ("x1", "x2", "x3")


def _make_results(x1, x2, x3, seed: int) -> GrangerAnalysisResults:
    """Build a one-output store with exact-OLS U/R snapshots."""
    # Independent generator: reusing the feature seed would reproduce x1
    # as the noise and drive the RSS to numerical zero.
    noise_rng = np.random.default_rng(10_000 + seed)
    x = np.column_stack([x1, x2, x3])
    y = (
        1.0
        + 0.4 * x1
        + 0.8 * x2
        + 1.5 * x3
        + 0.5 * noise_rng.normal(size=N_SAMPLES)
    )[:, None]
    ones = np.ones((N_SAMPLES, 1))

    results = GrangerAnalysisResults(
        effects=["y"],
        causes=["x2", "x3"],
        mask=np.array([[True, True, True]]),
        cause_slices={"x2": slice(1, 2), "x3": slice(2, 3)},
        y_true=y,
        X_true=x,
    )

    for cause, removed in (("x2", 1), ("x3", 2)):
        u_mask = np.array([[True, True, True]])
        r_mask = np.array([[True, True, True]])
        r_mask[0, removed] = False

        design_u = np.column_stack([ones, x])
        design_r = np.column_stack([ones, x[:, r_mask[0]]])

        beta_u = np.linalg.lstsq(design_u, y, rcond=None)[0]
        beta_r = np.linalg.lstsq(design_r, y, rcond=None)[0]

        weights_u = np.zeros((1, 3))
        weights_u[0, u_mask[0]] = beta_u[1:].ravel()
        weights_r = np.zeros((1, 3))
        weights_r[0, r_mask[0]] = beta_r[1:].ravel()

        results.register_snapshot(
            cause,
            "unrestricted",
            ModelSnapshot(
                predictions=design_u @ beta_u,
                weights=weights_u,
                active_slope_mask=u_mask,
                metadata={"coefficient_space": "raw"},
            ),
        )
        results.register_snapshot(
            cause,
            "restricted",
            ModelSnapshot(
                predictions=design_r @ beta_r,
                weights=weights_r,
                active_slope_mask=r_mask,
                metadata={"coefficient_space": "raw"},
            ),
        )

    return results


@pytest.fixture
def full_rank_results():
    """Independent features: structural and rank DOF agree."""
    rng = np.random.default_rng(11)
    return _make_results(
        rng.normal(size=N_SAMPLES),
        rng.normal(size=N_SAMPLES),
        rng.normal(size=N_SAMPLES),
        seed=11,
    )


@pytest.fixture
def collinear_results():
    """x2 duplicates x1: removing x2 removes no independent direction."""
    rng = np.random.default_rng(23)
    x1 = rng.normal(size=N_SAMPLES)
    return _make_results(
        x1,
        x1.copy(),  # exact collinearity with x1
        rng.normal(size=N_SAMPLES),
        seed=23,
    )


### DOF registry and strategies


def test_builtin_dof_methods_are_registered():
    available = available_dof()
    for name in ("structural", "active_support", "rank"):
        assert name in available


def test_structural_counts(full_rank_results):
    results = full_rank_results
    results.set_dof_method("structural")

    dimensions = results.dof_for("x3")

    np.testing.assert_array_equal(dimensions.model_df_u, [4.0])
    np.testing.assert_array_equal(dimensions.model_df_r, [3.0])
    np.testing.assert_array_equal(dimensions.restriction_df, [1.0])
    np.testing.assert_array_equal(dimensions.residual_df, [N_SAMPLES - 4.0])
    assert dimensions.status == ("ok",)
    assert not dimensions.heuristic


def test_rank_counts_match_structural_when_full_rank(full_rank_results):
    results = full_rank_results
    results.set_dof_method("rank", params={"rank_rcond": 1e-10})

    structural = full_rank_results.dof_for("x3")
    results.set_dof_method("structural")
    rank = results.dof_for("x3")
    results.set_dof_method("rank", params={"rank_rcond": 1e-10})

    np.testing.assert_array_equal(rank.restriction_df, structural.restriction_df)
    np.testing.assert_array_equal(rank.model_df_u, structural.model_df_u)
    assert rank.status == ("ok",)


def test_rank_detects_redundant_removed_column(collinear_results):
    results = collinear_results
    results.set_dof_method("rank")

    dimensions = results.dof_for("x2")

    np.testing.assert_array_equal(dimensions.model_df_u, [3.0])
    np.testing.assert_array_equal(dimensions.restriction_df, [0.0])
    assert dimensions.status == ("no_restrictions",)

    # Structural counting still reports one removed coefficient.
    results.set_dof_method("structural")
    assert results.dof_for("x2").restriction_df[0] == 1.0


def test_active_support_uses_unrestricted_support(full_rank_results):
    results = full_rank_results

    results.set_dof_method("active_support")
    default = results.dof_for("x3")
    assert default.restriction_df[0] == 1.0
    assert default.heuristic

    results.set_dof_method(
        "active_support",
        params={"active_support_tolerance": 1e6},
    )
    zeroed = results.dof_for("x3")
    np.testing.assert_array_equal(zeroed.restriction_df, [0.0])
    assert zeroed.status == ("no_restrictions",)


def test_dof_params_reach_the_context(full_rank_results):
    results = full_rank_results
    params = {"rank_rcond": 1e-10}
    results.set_dof_method("rank", params=params)

    assert results.dof_params == params

    matrix = results.evaluate("f_statistic", restore_removed=False)
    assert matrix.metadata["dof_params"] == params
    assert matrix.metadata["dof_method"] == "rank"


def test_invalid_dof_param_is_rejected(full_rank_results):
    results = full_rank_results
    results.set_dof_method("rank", params={"rank_rcond": 2.0})

    with pytest.raises(ResultsError):
        results.dof_for("x3")


def test_custom_fractional_strategy(full_rank_results):
    results = full_rank_results
    results.set_dof_method(fractional_dof)

    matrix = results.evaluate("f_statistic", restore_removed=False)

    assert matrix.metadata["n_restrictions"][0, 0] == 1.5
    assert matrix.metadata["residual_df"][0, 0] == N_SAMPLES - 2.5
    assert np.isfinite(matrix.data.iloc[0, 0])


def fractional_dof(context):
    """Example external strategy with fractional degrees of freedom."""
    size = len(context.effects)
    return DofResult(
        model_df_u=np.full(size, 2.5),
        model_df_r=np.full(size, 1.0),
        restriction_df=np.full(size, 1.5),
        residual_df=np.full(size, float(context.n_samples) - 2.5),
        status=("user_supplied",) * size,
        heuristic=True,
    )


def test_register_dof_alias_and_duplicate():
    def strategy(context):
        raise NotImplementedError

    register_dof("_test_dof_strategy", "_test_dof_alias")(strategy)

    assert resolve_dof(" _TEST_DOF_ALIAS ") is strategy

    with pytest.raises(Exception):
        register_dof("_test_dof_strategy")(strategy)


### Metric consistency on exact OLS fits


def test_wald_equals_qf(full_rank_results):
    results = full_rank_results
    results.set_dof_method("structural")

    f_value = results.evaluate(
        "f_statistic", restore_removed=False
    ).data.loc["y", "x3"]
    wald = results.evaluate(
        "wald", restore_removed=False
    ).data.loc["y", "x3"]
    q = results.dof_for("x3").restriction_df[0]

    assert np.isclose(wald, q * f_value, rtol=1e-6)


def test_projection_wald_equals_qf(full_rank_results):
    results = full_rank_results
    results.set_dof_method("rank")

    f_value = results.evaluate(
        "f_statistic", restore_removed=False
    ).data.loc["y", "x3"]
    projection = results.evaluate(
        "wald_projection", restore_removed=False
    ).data.loc["y", "x3"]
    q = results.dof_for("x3").restriction_df[0]

    assert np.isclose(projection, q * f_value, rtol=1e-6)


def test_likelihood_ratio_identity(full_rank_results):
    results = full_rank_results
    results.set_dof_method("structural")

    f_value = results.evaluate(
        "f_statistic", restore_removed=False
    ).data.loc["y", "x3"]
    lr = results.evaluate(
        "likelihood_ratio", restore_removed=False
    ).data.loc["y", "x3"]
    q = results.dof_for("x3").restriction_df[0]
    df = results.dof_for("x3").residual_df[0]

    expected = results.n_samples * np.log(1.0 + q * f_value / df)
    assert np.isclose(lr, expected, rtol=1e-6)


def test_p_values_are_consistent_and_significant(full_rank_results):
    results = full_rank_results
    results.set_dof_method("structural")

    p_f = results.evaluate(
        "f_test_p_value", restore_removed=False
    ).data.loc["y", "x3"]
    p_wald = results.evaluate(
        "wald_test_p_value", restore_removed=False
    ).data.loc["y", "x3"]
    p_lr = results.evaluate(
        "likelihood_ratio_p_value", restore_removed=False
    ).data.loc["y", "x3"]
    p_projection = results.evaluate(
        "wald_projection_p_value", restore_removed=False
    ).data.loc["y", "x3"]

    for p_value in (p_f, p_wald, p_lr, p_projection):
        assert np.isfinite(p_value)
        assert 0.0 <= p_value <= 1.0

    # x3 has a strong true effect: every test should reject the null.
    assert p_f < 0.01
    assert p_wald < 0.01
    assert p_lr < 0.01
    assert p_projection < 0.01


def test_projection_rejects_incompatible_dof(full_rank_results):
    results = full_rank_results
    results.set_dof_method(fractional_dof)

    matrix = results.evaluate(
        "wald_projection", restore_removed=False
    )

    assert np.isnan(matrix.data.loc["y", "x3"])
    assert matrix.metadata["status"][0, 1] == (
        "dof_incompatible_with_projection"
    )


### Collinearity behavior


def test_coefficient_wald_unavailable_for_rank_deficient_design(collinear_results):
    results = collinear_results
    results.set_dof_method("structural")

    matrix = results.evaluate("wald", restore_removed=False)

    assert np.isnan(matrix.data.loc["y", "x2"])
    assert matrix.metadata["status"][0, 0] == (
        "coefficient_design_rank_deficient"
    )


def test_rank_wald_reports_no_additional_directions(collinear_results):
    results = collinear_results
    results.set_dof_method("rank")

    matrix = results.evaluate("wald_projection", restore_removed=False)

    assert np.isnan(matrix.data.loc["y", "x2"])
    assert matrix.metadata["status"][0, 0] == (
        "no additional retained directions"
    )
    # x3 remains testable in the same matrix.
    assert np.isfinite(matrix.data.loc["y", "x3"])


def test_f_statistic_unavailable_with_zero_rank_restrictions(collinear_results):
    results = collinear_results
    results.set_dof_method("rank")

    statistic = results.evaluate("f_statistic", restore_removed=False)
    p_value = results.evaluate(
        "f_statistic_p_value", restore_removed=False
    )

    assert np.isnan(statistic.data.loc["y", "x2"])
    assert statistic.metadata["status"][0, 0] == (
        "invalid_restriction_df"
    )
    assert np.isnan(p_value.data.loc["y", "x2"])
    # The p-value layer only sees a missing statistic; the DOF-level
    # reason is preserved in the statistic status.
    assert p_value.metadata["status"][0, 0] == (
        "statistic_unavailable"
    )
    assert p_value.metadata["statistic_status"][0, 0] == (
        "invalid_restriction_df"
    )


def test_structural_f_reports_zero_improvement(collinear_results):
    results = collinear_results
    results.set_dof_method("structural")

    p_value = results.evaluate(
        "f_test_p_value", restore_removed=False
    ).data.loc["y", "x2"]

    # Removing a redundant column cannot improve the fit.
    assert p_value > 0.5


### Unavailable-test policies


def test_unavailable_policies_for_missing_tests(collinear_results):
    results = collinear_results
    results.set_dof_method("rank")

    zeroed = results.result(
        metric="wald_projection_p_value",
        unavailable_policy="zero",
    )
    assert zeroed.loc["y", "x2"] == 0

    preserved = results.result(
        metric="wald_projection_p_value",
        unavailable_policy="preserve",
    )
    assert pd.isna(preserved.loc["y", "x2"])

    with pytest.raises(ValueError):
        results.result(
            metric="wald_projection_p_value",
            unavailable_policy="raise",
        )


def test_status_metadata_is_exposed(collinear_results):
    results = collinear_results
    results.set_dof_method("rank")

    matrix = results.evaluate(
        "wald_projection_p_value", restore_removed=False
    )

    # The p-value layer reports "statistic_unavailable"; the statistic-level
    # reason is preserved in "statistic_status".
    status = pd.DataFrame(
        matrix.metadata["statistic_status"],
        index=matrix.metadata["effective_effects"],
        columns=matrix.metadata["effective_causes"],
    )
    assert status.loc["y", "x2"] == "no additional retained directions"
    assert status.loc["y", "x3"] == "ok"
    assert matrix.metadata["status"][0, 0] == "statistic_unavailable"
