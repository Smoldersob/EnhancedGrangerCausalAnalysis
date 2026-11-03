"""Numerical integration checks for bias, scaling, masks and Wald/F consistency.

Run from the repository root:
    python -m pytest enhanced_granger_analysis/tests/test_wald_f_bias_consistency.py -v

Requires the version_3_ runner API with train_only_unrestricted.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi2, f as f_distribution

from ..backends.numpy.numpy_backend import NumpyBackendStrategy
from ..core.components.constraint_spec import PreparedConstraintData
from ..core.models.input_spec import ModelInputSpec
from ..execution.runner import AnalysisRunner, AnalysisRunProgress
from ..results.granger_results import GrangerAnalysisResults
from ..results.statistics import estimate_masked_weight_covariance


@pytest.fixture(scope="module")
def sm():
    # This is a required reference dependency, not an optional skipped check.
    import statsmodels.api as sm_api
    return sm_api


def make_data(n_outputs, use_bias, scaled):
    """Prepare deterministic full-rank data with nonzero predictor means."""
    rng = np.random.default_rng(38177)
    n = 360
    x = rng.normal(size=(n, 4))
    x[:, 1] += 0.35 * x[:, 0]
    x = x * np.array([0.2, 3.0, 8.0, 0.7]) + np.array([1.0, -2.0, 3.0, 0.5])
    mask = np.array([
        [True, True, False, False],
        [False, True, True, True],
        [True, True, True, False],
    ])[:n_outputs]
    slopes = np.array([
        [0.4, -0.18, 0.0, 0.0],
        [0.0, 0.12, 0.07, -0.6],
        [-0.25, 0.2, -0.05, 0.0],
    ])[:n_outputs]
    intercept = np.array([1.3, -2.5, 0.7])[:n_outputs] if use_bias else np.zeros(n_outputs)
    y = x @ slopes.T + intercept + rng.normal(size=(n, n_outputs)) * np.array([0.4, 1.7, 0.8])[:n_outputs]
    x_scale = x.std(axis=0) if scaled else np.ones(4)
    y_scale = y.std(axis=0) if scaled else np.ones(n_outputs)
    # Center only when an intercept is part of the model. With no intercept,
    # pure rescaling preserves the original zero-intercept model family.
    x_offset = x.mean(axis=0) if scaled and use_bias else np.zeros(4)
    y_offset = y.mean(axis=0) if scaled and use_bias else np.zeros(n_outputs)
    xm = (x - x_offset) / x_scale
    ym = (y - y_offset) / y_scale
    prepared = SimpleNamespace(
        X_raw=x, y_raw=y, X_scaled=xm, y_scaled=ym,
        effects=tuple(f"y{j}" for j in range(n_outputs)),
        causes=("a", "b"), tested_causes=("a", "b"),
        cause_slices={"a": slice(0, 2), "b": slice(2, 4)},
        feature_lags=(1, 2, 1, 2), base_mask=mask.copy(),
        y_scaler=SimpleNamespace(inverse_transform=lambda value: np.asarray(value) * y_scale + y_offset),
        indices_for=lambda *args, **kwargs: np.arange(n, dtype=np.intp),
    )
    return prepared, mask


def run_case(prepared, mask, use_bias, only_u=False):
    """Fit the real masked NumPy adapter through the actual AnalysisRunner."""
    strategy = NumpyBackendStrategy()
    constraint_data = PreparedConstraintData(
        output_names=prepared.effects, n_features=4, mask=mask.copy(), rules=(),
    )
    constraint = strategy.build_constraint("mask", data=constraint_data)
    spec = ModelInputSpec(
        n_features=4, n_outputs=len(prepared.effects), causes=prepared.causes,
        effects=prepared.effects, cause_slices=prepared.cause_slices,
        feature_lags=prepared.feature_lags,
    )
    model = strategy.build_model(
        "conditional",
        components={"constraint": constraint, "regularizer": None, "callbacks": []},
        config={"input_spec": spec, "solver": "restricted_lstsq", "use_bias": use_bias,
                "epochs": 3, "batch_size": None, "learning_rate": 1.0,
                "tol": 1e-12, "verbose": 0},
    )
    results = GrangerAnalysisResults(
        effects=prepared.effects, causes=prepared.tested_causes, mask=mask.copy(),
        cause_slices=prepared.cause_slices, y_true=prepared.y_raw, X_true=prepared.X_raw,
    )
    results.set_dof_method("structural")
    # The runner only needs prepared_data for model-coordinate Wald evaluation.
    # Restoration is deliberately disabled in every metric call below.
    results.set_preparation_context(SimpleNamespace(prepared_data=prepared))
    runner = AnalysisRunner()
    runner.run(model=model, prepared=prepared,
               progress=AnalysisRunProgress(results=results), backend="numpy",
               train_only_unrestricted=only_u)
    assert model.solver_used_ == "restricted_lstsq"
    return results


def matrix(results, name):
    return results.evaluate(name, restore_removed=False, force=True).data.to_numpy(dtype=float)


def reference_fit(sm, x, y, mask, use_bias):
    """Fit each output independently only to form a reference calculation."""
    fits = []
    for j, active in enumerate(mask):
        design = x[:, active]
        if use_bias:
            design = np.column_stack([np.ones(len(x)), design])
        if design.shape[1] == 0:
            # The zero-intercept, zero-slope model predicts zero identically.
            rss = float(y[:, j] @ y[:, j])
            n = len(y)
            fits.append(SimpleNamespace(
                ssr=rss, df_resid=float(n),
                llf=-0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(rss / n)),
                fittedvalues=np.zeros(n),
            ))
        else:
            fits.append(sm.OLS(y[:, j], design).fit())
    return fits


@pytest.mark.parametrize("n_outputs", [2, 3])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_wald_f_bias_and_reference(sm, n_outputs, use_bias, scaled):
    prepared, mask = make_data(n_outputs, use_bias, scaled)
    results = run_case(prepared, mask, use_bias)
    actual_f = matrix(results, "f_statistic")
    actual_w = matrix(results, "wald_test")
    actual_fp = matrix(results, "f_statistic_p_value")
    actual_wp = matrix(results, "wald_test_p_value")
    actual_lr = matrix(results, "likelihood_ratio")
    actual_lrp = matrix(results, "likelihood_ratio_p_value")
    u_fits = reference_fit(sm, prepared.X_scaled, prepared.y_scaled, mask, use_bias)
    expected_k = mask.sum(axis=1) + int(use_bias)
    for col, cause in enumerate(prepared.tested_causes):
        u = results.get_snapshot(cause, "unrestricted")
        r = results.get_snapshot(cause, "restricted")
        assert u.bias_enabled == use_bias
        assert r.bias_enabled == use_bias
        np.testing.assert_array_equal(u.active_slope_mask, mask)
        rm = mask.copy()
        rm[:, prepared.cause_slices[cause]] = False
        np.testing.assert_array_equal(r.active_slope_mask, rm)
        q = (mask & ~rm).sum(axis=1)
        np.testing.assert_array_equal(results.n_restrictions_for(cause), q)
        np.testing.assert_array_equal(results.n_unrestricted_parameters(cause), expected_k)
        r_fits = reference_fit(sm, prepared.X_scaled, prepared.y_scaled, rm, use_bias)
        for j, (uf, rf) in enumerate(zip(u_fits, r_fits)):
            if q[j] == 0:
                assert np.isnan(actual_f[j, col])
                assert np.isnan(actual_w[j, col])
                assert np.isnan(actual_fp[j, col])
                assert np.isnan(actual_wp[j, col])
                continue
            ref_f, ref_fp, ref_q = uf.compare_f_test(rf)
            ref_lr, ref_lrp, _ = uf.compare_lr_test(rf)
            assert ref_q == q[j]
            np.testing.assert_allclose(actual_f[j, col], ref_f, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(actual_w[j, col], q[j] * ref_f, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(actual_fp[j, col], ref_fp, rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(actual_wp[j, col], chi2.sf(q[j] * ref_f, q[j]), rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(actual_lr[j, col], ref_lr, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(actual_lrp[j, col], ref_lrp, rtol=1e-8, atol=1e-12)
            p_wald_as_f = f_distribution.sf(actual_w[j, col] / q[j], q[j], len(prepared.X_raw) - expected_k[j])
            np.testing.assert_allclose(p_wald_as_f, actual_fp[j, col], rtol=1e-8, atol=1e-12)
            for alpha in [0.01, 0.05]:
                assert (p_wald_as_f < alpha) == (actual_fp[j, col] < alpha)


@pytest.mark.parametrize("use_bias", [False, True])
def test_scaling_does_not_change_statistics(sm, use_bias):
    raw_data, mask = make_data(3, use_bias, False)
    scaled_data, _ = make_data(3, use_bias, True)
    raw = run_case(raw_data, mask, use_bias)
    scaled = run_case(scaled_data, mask, use_bias)
    for metric in ["f_statistic", "f_statistic_p_value", "wald_test", "wald_test_p_value",
                   "likelihood_ratio", "likelihood_ratio_p_value"]:
        np.testing.assert_allclose(matrix(raw, metric), matrix(scaled, metric),
                                   rtol=1e-8, atol=1e-9, equal_nan=True,
                                   err_msg=f"Scaling changed {metric}; bias={use_bias}")


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_restricted_refit_does_not_change_wald(sm, use_bias, scaled):
    prepared, mask = make_data(3, use_bias, scaled)
    full = run_case(prepared, mask, use_bias, only_u=False)
    only_u = run_case(prepared, mask, use_bias, only_u=True)
    for cause in prepared.tested_causes:
        uf = full.get_snapshot(cause, "unrestricted")
        uo = only_u.get_snapshot(cause, "unrestricted")
        np.testing.assert_allclose(uf.weights, uo.weights, rtol=0, atol=0)
        np.testing.assert_allclose(uf.predictions, uo.predictions, rtol=0, atol=0)
        assert only_u.get_snapshot(cause, "restricted").metadata["fit_performed"] is False
    for metric in ["wald_test", "wald_test_p_value"]:
        np.testing.assert_allclose(matrix(full, metric), matrix(only_u, metric),
                                   rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_covariance_accounts_for_intercept(sm, use_bias, scaled):
    prepared, mask = make_data(3, use_bias, scaled)
    fits = reference_fit(sm, prepared.X_scaled, prepared.y_scaled, mask, use_bias)
    predictions = np.column_stack([fit.fittedvalues for fit in fits])
    cov = estimate_masked_weight_covariance(
        x_train=prepared.X_scaled, y_true=prepared.y_scaled, y_pred=predictions,
        coefficient_mask=mask, fit_intercept=use_bias,
    )
    for j, fit in enumerate(fits):
        active = np.flatnonzero(mask[j])
        expected = np.asarray(fit.cov_params())
        if use_bias:
            expected = expected[1:, 1:]
        np.testing.assert_allclose(cov[j][np.ix_(active, active)], expected,
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_array_equal(cov[j][~mask[j]], 0.)
        np.testing.assert_array_equal(cov[j][:, ~mask[j]], 0.)
