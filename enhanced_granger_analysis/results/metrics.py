from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.exceptions import ResultsError
from .causality_matrix import CausalityMatrix
from .statistics import (
    estimate_masked_weight_covariance,
    f_test_value,
    likelihood_ratio_test_value,
    p_value_from_f_test,
    p_value_from_likelihood_ratio_test,
    p_value_from_wald_test,
    wald_test_value,
)

Metric = Callable[..., CausalityMatrix]

_METRICS: dict[str, Metric] = {}

def register_metric(*names: str):
    """Register aliases used by ``GrangerAnalysisResults.evaluate``."""
    def decorator(func: Metric) -> Metric:
        for name in names:
            if name in _METRICS:
                raise ResultsError(f"Metric '{name}' is already registered.")
            _METRICS[name] = func
        return func

    return decorator


def resolve_metric(metric: str | Metric) -> Metric:
    """Resolve a registered metric name or accept a custom metric callable."""
    if callable(metric):
        return metric

    normalized = str(metric).strip().lower()
    try:
        return _METRICS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_METRICS))
        raise ResultsError(f"Unknown metric '{metric}'. Available metrics: {available}.") from exc


def _statistic_matrix(results: Any, name: str) -> CausalityMatrix:
    """Reuse a cached statistic or evaluate it once without restoration."""
    if name in results._metric_cache:
        return results._metric_cache[name]
    return results.evaluate(name, restore_removed=False)


def _p_value_from_statistic(results: Any, statistic_name: str) -> CausalityMatrix:
    """Convert one named statistic matrix to its matching p-value matrix."""
    statistic = _statistic_matrix(results, statistic_name)
    stat_values = statistic.data.to_numpy(dtype=np.float64)
    q_matrix = np.empty_like(stat_values, dtype=np.int_)
    for cause_idx, cause in enumerate(results.causes):
        q_matrix[:, cause_idx] = results.n_restrictions_for(cause)

    if statistic_name == "f_statistic":
        residual_df = np.full_like(
            stat_values,
            max(results.n_samples - results.n_unrestricted_parameters(), 1),
            dtype=np.int_,
        )
        p_values = p_value_from_f_test(stat_values, q_matrix, residual_df)
        name = "f_statistic_p_value"
    elif statistic_name == "wald_test":
        p_values = p_value_from_wald_test(stat_values, q_matrix)
        name = "wald_test_p_value"
    else:
        p_values = p_value_from_likelihood_ratio_test(stat_values, q_matrix)
        name = "likelihood_ratio_p_value"

    return results.make_matrix(
        p_values,
        name=name,
        metadata={"removed_signal_value": 1.0},
    )


def _p_value_metric_name(results: Any) -> str:
    """Find a registered p-value companion for an evaluated statistic.

    The companion convention is ``<statistic_name>_p_value``. This keeps the
    meta-metric independent of the available statistical tests.
    """
    for cache_key in results._metric_cache:
        if not isinstance(cache_key, str):
            continue
        candidate = f"{cache_key}_p_value"
        try:
            resolve_metric(candidate)
        except ResultsError:
            continue
        return candidate

    default_name = "f_statistic_p_value"
    try:
        resolve_metric(default_name)
    except ResultsError as exc:
        raise ResultsError(
            "No registered metric has a <metric>_p_value companion."
        ) from exc
    return default_name


@register_metric("f_statistic", "f")
def f_statistic(results: Any) -> CausalityMatrix:
    """Compute the Granger F statistic per output and tested cause.

    Assumes the reference model is restricted relative to the base model.
    The restriction count may use structural or active-support degrees of
    freedom, according to ``results.dof_method``.
    """
    base_rss = results.base_rss_per_output()
    values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)

    for cause_col, cause in enumerate(results.causes):
        ref_rss = results.reference_rss_per_output(cause)
        q = results.n_restrictions_for(cause)
        k_u = results.n_unrestricted_parameters()
        values[:, cause_col] = f_test_value(
            error_ref=ref_rss,
            error_base=base_rss,
            n_restrictions=q,
            n_unrestricted_parameters=k_u,
            n_samples=results.n_samples,
        )

    return results.make_matrix(values, name="f_statistic", metadata={"removed_signal_value": 0.0})


@register_metric(
    "f_statistic_p_value",
    "f_test_p_value",
    "f_p_value",
    "p_value_f",
)
def f_test_p_value(results: Any) -> CausalityMatrix:
    """Convert F statistics to upper-tail p-values per output and cause."""
    return _p_value_from_statistic(results, "f_statistic")


@register_metric("wald_test", "wald", "wald_statistic")
def wald_test(results: Any) -> CausalityMatrix:
    """Compute Wald statistics from base weights and per-output OLS covariance.

    The design matrix, target values, predictions, and coefficient mask must
    describe the same multi-output model. Structurally inactive coefficients
    are excluded from each tested block.
    """
    if results.base_snapshot is None or results.base_snapshot.weights is None or results.X_true.size == 0:
        values = np.full((len(results.effects), len(results.causes)), np.nan, dtype=np.float64)
        return results.make_matrix(values, name="wald_test", metadata={"removed_signal_value": 0.0})

    covariance = estimate_masked_weight_covariance(
        x_train=results.X_true,
        y_true=results.y_true,
        y_pred=results.base_snapshot.predictions,
        coefficient_mask=results.mask,
        fit_intercept=False,
    )
    weights = np.asarray(results.base_snapshot.weights, dtype=np.float64)
    if weights.shape[1] == len(results.effects):
        weights = weights.T
    values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)
    for cause_col, cause in enumerate(results.causes):
        start = int(results.col_offsets[cause_col])
        end = int(results.col_offsets[cause_col + 1])
        values[:, cause_col] = wald_test_value(
            weights=weights,
            covariance=covariance,
            coefficient_mask=results.mask,
            start=start,
            end=end,
        )
    return results.make_matrix(values, name="wald_test", metadata={"removed_signal_value": 0.0})


@register_metric(
    "wald_test_p_value",
    "wald_statistic_p_value",
    "wald_p_value",
    "p_value_wald",
)
def wald_p_value(results: Any) -> CausalityMatrix:
    """Convert Wald chi-square statistics to p-values per output and cause."""
    return _p_value_from_statistic(results, "wald_test")


@register_metric("likelihood_ratio", "likelihood_ratio_test", "lr", "lr_test")
def likelihood_ratio_test(results: Any) -> CausalityMatrix:
    """Compute the RSS-based likelihood-ratio statistic per output and cause.

    The reference and base predictions must use the same observations and
    output ordering. The statistic is evaluated against the base RSS.
    """
    base_rss = results.base_rss_per_output()
    values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)
    for cause_col, cause in enumerate(results.causes):
        ref_rss = results.reference_rss_per_output(cause)
        values[:, cause_col] = likelihood_ratio_test_value(
            error_ref=ref_rss,
            error_base=base_rss,
            n_samples=results.n_samples,
        )
    return results.make_matrix(values, name="likelihood_ratio", metadata={"removed_signal_value": 0.0})


@register_metric(
    "likelihood_ratio_p_value",
    "likelihood_ratio_test_p_value",
    "lr_p_value",
    "p_value_lr",
    "p_value_likelihood_ratio",
)
def likelihood_ratio_p_value(results: Any) -> CausalityMatrix:
    """Convert likelihood-ratio chi-square statistics to p-values."""
    return _p_value_from_statistic(results, "likelihood_ratio")


@register_metric("p_value")
def p_value(results: Any) -> CausalityMatrix:
    """Convert the best available cached test statistic to p-values.

    A previously evaluated statistic is paired with a registered metric named
    ``<statistic_name>_p_value``. If none is cached, F is used as the default.
    """
    p_value_name = _p_value_metric_name(results)
    return resolve_metric(p_value_name)(results)


@register_metric("sign")
def sign_metric(results: Any) -> CausalityMatrix:
    """Return the sign of the summed base coefficients per output and cause."""
    sign_matrix = np.zeros((len(results.effects), len(results.causes)), dtype=np.float64)
    for cause_idx, cause in enumerate(results.causes):
        weights = results.base_snapshot.weights if results.base_snapshot is not None else None
        if weights is None:
            continue
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape[1] == len(results.effects):
            weights_arr = weights_arr.T
        start = int(results.col_offsets[cause_idx])
        end = int(results.col_offsets[cause_idx + 1])
        block = weights_arr[:, start:end]
        sums = block.sum(axis=1)
        sign_matrix[:, cause_idx] = np.sign(sums)
        sign_matrix[np.abs(sums) < 1e-12, cause_idx] = 1.0
    return results.make_matrix(sign_matrix, name="sign", metadata={"removed_signal_value": 0.0})


@register_metric("base_mse")
def base_mse(results: Any) -> CausalityMatrix:
    """Return unrestricted-model MSE per output, repeated for each cause."""
    if results.base_snapshot is None:
        values = np.full((len(results.effects), len(results.causes)), np.nan, dtype=np.float64)
        return results.make_matrix(values, name="base_mse")
    mse = results.base_rss_per_output() / max(results.n_samples, 1)
    values = np.repeat(mse[:, np.newaxis], len(results.causes), axis=1)
    return results.make_matrix(values, name="base_mse")


@register_metric("base_rss")
def base_rss(results: Any) -> CausalityMatrix:
    """Return unrestricted-model RSS per output, repeated for each cause."""
    if results.base_snapshot is None:
        values = np.full((len(results.effects), len(results.causes)), np.nan, dtype=np.float64)
        return results.make_matrix(values, name="base_rss")
    rss = results.base_rss_per_output()
    values = np.repeat(rss[:, np.newaxis], len(results.causes), axis=1)
    return results.make_matrix(values, name="base_rss")
