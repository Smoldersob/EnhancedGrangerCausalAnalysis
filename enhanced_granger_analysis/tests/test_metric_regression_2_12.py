"""Numerical regression tests for version_3_ metric fixes.
Run from the repository root: python -m pytest test_metric_regressions.py -q
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import f, chi2

from ..core.exceptions import ResultsError
from ..results.granger_results import (
    GrangerAnalysisResults,
    ModelSnapshot,
)
from ..results.statistics import f_test_value, p_value_from_f_test
from ..execution.runner import AnalysisRunner


def test_zero_restrictions_are_local_to_one_output():
    value = f_test_value(np.array([20., 10.]), np.array([10., 10.]),
                         np.array([1, 0]), np.array([3, 3]), 100)
    assert value[0] == pytest.approx(97.)
    assert np.isnan(value[1])


def test_exact_unrestricted_fit_has_zero_p_value():
    value = f_test_value(np.array([10.]), np.array([0.]),
                         np.array([1]), np.array([2]), 20)
    assert np.isposinf(value[0])
    np.testing.assert_array_equal(p_value_from_f_test(value, np.array([1]), np.array([18])), [0.])

def test_worse_unrestricted_fit_is_neutral():
    value = f_test_value(
        error_ref=np.array([1.0]),
        error_base=np.array([2.0]),
        n_restrictions=np.array([1]),
        n_unrestricted_parameters=np.array([2]),
        n_samples=20,
    )

    np.testing.assert_array_equal(value, [0.0])

    p_value = p_value_from_f_test(
        value,
        np.array([1]),
        np.array([18]),
    )
    np.testing.assert_array_equal(p_value, [1.0])

def make_results(n_outputs=3, scaled=False):
    rng = np.random.default_rng(381)
    n = 240
    x_model = rng.normal(size=(n, 4))
    u_mask = np.array([[1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=bool)[:n_outputs]
    beta = np.array([[.16, -.12, 0, .5], [.12, .09, .3, -.2], [.1, -.13, .4, 0]])[:n_outputs]
    y_model = x_model @ beta.T + rng.normal(size=(n, n_outputs))
    x_scale, x_offset = np.array([2., 7., .2, 4.]), np.array([10., -3., 2., 8.])
    y_scale, y_offset = np.arange(1, n_outputs + 1) * 3., np.arange(n_outputs) + 11.
    x_raw = x_model * x_scale + x_offset if scaled else x_model
    y_raw = y_model * y_scale + y_offset if scaled else y_model
    effects = [f"y{i}" for i in range(n_outputs)]
    results = GrangerAnalysisResults(effects=effects, causes=["a", "b"], mask=u_mask,
        cause_slices={"a": slice(0, 1), "b": slice(1, 2)}, X_true=x_raw, y_true=y_raw)
    prepared = SimpleNamespace(
            X_scaled=x_model,
            y_scaled=y_model,
            tested_causes=list(results.causes),
            y_scaler=SimpleNamespace(
                inverse_transform=lambda y: y * y_scale + y_offset
            ),
        )
    if scaled:
        results.preparation_context = SimpleNamespace(prepared_data=prepared)
    sm = pytest.importorskip("statsmodels.api")
    reference = {}
    for cause, omitted in [("a", 0), ("b", 1)]:
        r_mask = u_mask.copy()
        r_mask[:, omitted] = False
        preds, coeffs, fits = {}, {}, {}
        for role, mask in [("unrestricted", u_mask), ("restricted", r_mask)]:
            pred = np.empty_like(y_model)
            weights = np.zeros_like(beta)
            models = []
            for j in range(n_outputs):
                active = np.flatnonzero(mask[j])
                design = np.column_stack([np.ones(n), x_model[:, active]])
                fit = sm.OLS(y_model[:, j], design).fit()
                pred[:, j] = fit.predict(design)
                weights[j, active] = fit.params[1:]
                models.append(fit)
            preds[role], coeffs[role], fits[role] = pred, weights, models
            if scaled:
                outcome = SimpleNamespace(predictions=pred, weights=[weights.T], timings={},
                    active_slope_mask=mask, bias_enabled=True)
                AnalysisRunner.store_fit_outcome(prepared=prepared, results=results,
                    cause=cause, outcome=outcome, backend="numpy", role=role)
            else:
                results.register_snapshot(
                    cause=cause,
                    role=role,
                    snapshot=ModelSnapshot(
                        predictions=pred.copy(),
                        weights=weights.copy(),
                        active_slope_mask=mask.copy(),
                        bias_enabled=True,
                        metadata={
                            "coefficient_space": "raw",
                            "fit_performed": True,
                        },
                    ),
                )
        reference[cause] = (fits["unrestricted"], fits["restricted"])
    return results, reference


@pytest.mark.parametrize("n_outputs", [2, 3])
@pytest.mark.parametrize("scaled", [False, True])
def test_metrics_match_independent_ols(n_outputs, scaled):
    results, reference = make_results(n_outputs, scaled)
    actual = {name: results.evaluate(name, restore_removed=False).data.to_numpy()
        for name in ["f_statistic", "f_statistic_p_value", "likelihood_ratio",
                     "likelihood_ratio_p_value", "wald_test", "wald_test_p_value"]}
    for col, cause in enumerate(results.causes):
        u_models, r_models = reference[cause]
        for j, (u, r) in enumerate(zip(u_models, r_models)):
            f_value, f_p, q = u.compare_f_test(r)
            lr_value, lr_p, _ = u.compare_lr_test(r)
            assert actual["f_statistic"][j, col] == pytest.approx(f_value, rel=1e-9, abs=1e-10)
            assert actual["f_statistic_p_value"][j, col] == pytest.approx(f_p, rel=1e-9, abs=1e-12)
            assert actual["likelihood_ratio"][j, col] == pytest.approx(lr_value, rel=1e-9, abs=1e-10)
            assert actual["likelihood_ratio_p_value"][j, col] == pytest.approx(lr_p, rel=1e-9, abs=1e-12)
            assert actual["wald_test"][j, col] == pytest.approx(q * f_value, rel=1e-9, abs=1e-10)
            assert actual["wald_test_p_value"][j, col] == pytest.approx(chi2.sf(q * f_value, q), rel=1e-9, abs=1e-12)


def test_active_support_parameter_counts():
    results, _ = make_results(2)

    structural = results.n_unrestricted_parameters("a").copy()
    snapshot = results.get_snapshot("a", "unrestricted")

    weights = snapshot.weights.copy()
    weights[0, 0] = 0.0

    results.register_snapshot(
        cause="a",
        role="unrestricted",
        snapshot=ModelSnapshot(
            predictions=snapshot.predictions.copy(),
            weights=weights,
            active_slope_mask=snapshot.active_slope_mask.copy(),
            bias_enabled=snapshot.bias_enabled,
            metadata=dict(snapshot.metadata),
        ),
    )

    results.set_dof_method("active_support")

    supported = results.n_unrestricted_parameters("a")

    assert supported[0] == structural[0] - 1
    assert supported[1] == structural[1]
