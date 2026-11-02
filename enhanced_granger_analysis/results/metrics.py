from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.registry import Registry
from ..core.exceptions import ResultsError, RegistryError
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

_METRICS: Registry[Metric] = Registry(label="metric")


def register_metric(*names: str) -> Callable[[Metric], Metric]:
    """Register a metric under a canonical name and optional aliases."""
    def decorator(func: Metric) -> Metric:
        if not callable(func):
            raise ResultsError("A registered metric must be callable.")

        try:
            return _METRICS.decorator(*names)(func)
        except RegistryError as exc:
            raise ResultsError(str(exc)) from exc

    return decorator


def resolve_metric(metric: str | Metric) -> Metric:
    """Resolve a metric name or accept a custom callable."""
    if callable(metric):
        return metric

    try:
        return _METRICS.resolve(metric)
    except RegistryError as exc:
        raise ResultsError(str(exc)) from exc


def available_metrics(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """List registered metrics without evaluating them."""
    return _METRICS.available(include_aliases=include_aliases)

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
        residual_df = np.empty_like(stat_values, dtype=np.int_)
        for col, cause in enumerate(results.causes):
            residual_df[:, col] = (
                results.n_samples - results.n_unrestricted_parameters(cause)
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
    values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)

    for cause_col, cause in enumerate(results.causes):
        base_rss = results.unrestricted_rss_per_output(cause)
        ref_rss = results.reference_rss_per_output(cause)
        q = results.n_restrictions_for(cause)
        k_u = results.n_unrestricted_parameters(cause)
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
def wald_test(results):
    """Evaluate slope restrictions in a consistent coefficient coordinate system."""
    values = np.full((len(results.effects), len(results.causes)), np.nan)
    covariance_cache = {}
    for col, cause in enumerate(results.causes):
        u = results.get_snapshot(cause, "unrestricted")
        r = results.get_snapshot(cause, "restricted")
        um = results._require_snapshot_mask(u)
        rm = results._require_snapshot_mask(r)
        if um.shape != rm.shape or np.any(rm & ~um):
            raise ResultsError("Restricted mask must be a subset of the unrestricted mask")
        if u.bias_enabled != r.bias_enabled:
            raise ResultsError("Compared snapshots must use matching intercept settings")
        if u.weights is None:
            continue
        tested = um & ~rm
        if results.dof_method == "active_support":
            tested &= np.abs(u.weights) > results.active_support_tolerance
        if not tested.any():
            continue
        space = u.metadata.get("coefficient_space")
        if space == "model":
            if results.preparation_context is None:
                raise ResultsError("Model-space Wald coefficients require preparation context")
            prepared = results.preparation_context.prepared_data
            x = np.asarray(prepared.X_scaled, dtype=float)
            y = np.asarray(prepared.y_scaled, dtype=float)
            if "model_predictions" not in u.metadata:
                raise ResultsError("Snapshot lacks model-space predictions; refit the analysis")
            predictions = np.asarray(u.metadata["model_predictions"], dtype=float)
        elif space == "raw" or (space is None and results.preparation_context is None):
            x, y, predictions = results.X_true, results.y_true, u.predictions
        else:
            raise ResultsError("Ambiguous coefficient scale in legacy snapshot; refit the analysis")
        key = id(u)
        if key not in covariance_cache:
            covariance_cache[key] = estimate_masked_weight_covariance(
                x_train=x, y_true=y, y_pred=predictions,
                coefficient_mask=um, fit_intercept=u.bias_enabled,
            )
        values[:, col] = wald_test_value(
            weights=results._weights_as_feature_output(u.weights),
            covariance=covariance_cache[key], coefficient_mask=um,
            start=0, end=um.shape[1], tested_mask=tested,
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
    values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)
    for cause_col, cause in enumerate(results.causes):
        base_rss = results.unrestricted_rss_per_output(cause)
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
    """Return coefficient-sum signs for effective effects and tested causes."""
    values = np.zeros(
        (len(results.effects), len(results.causes)),
        dtype=np.float64,
    )

    if not results.causes:
        return results.make_matrix(
            values,
            name="sign",
            metadata={"removed_signal_value": 0.0},
        )

    for cause_index, cause in enumerate(results.causes):
        snapshot = results._snapshot_for(cause, "unrestricted")
        if snapshot.weights is None or snapshot.active_slope_mask is None:
            continue
        weights = results._weights_as_feature_output(snapshot.weights)
        if weights.shape[0] != len(results.effects):
            raise ValueError(
                f"Weight shape {weights.shape} is incompatible with "
                f"{len(results.effects)} effects"
            )
        restricted = results._snapshot_for(cause, "restricted")
        tested_mask = snapshot.active_slope_mask & ~restricted.active_slope_mask
        if cause not in results.cause_slices:
            raise ValueError(
                f"Missing feature slice for tested cause: {cause!r}"
            )

        block = results.cause_slices[cause]
        if not isinstance(block, slice):
            raise ValueError(
                f"Expected slice for cause {cause!r}; "
                f"got {type(block).__name__}"
            )

        if (
            block.start is None
            or block.stop is None
            or block.step not in (None, 1)
            or not 0 <= block.start <= block.stop <= weights.shape[1]
        ):
            raise ValueError(
                f"Invalid feature slice for cause {cause!r}: {block!r}; "
                f"weights have {weights.shape[1]} feature columns"
            )

        sums = np.where(
            tested_mask[:, block],
            weights[:, block],
            0.0,
        ).sum(axis=1)
        signs = np.sign(sums)

        # Preserve the existing convention for near-zero coefficient sums.
        signs[np.abs(sums) < 1e-12] = 1.0
        values[:, cause_index] = signs

    return results.make_matrix(
        values,
        name="sign",
        metadata={"removed_signal_value": 0.0},
    )

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
