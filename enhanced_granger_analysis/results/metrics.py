from  __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import warnings

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
	rank_wald_test_value
)
from .granger_results import GrangerAnalysisResults

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


@register_metric("f_statistic", "f")
def f_statistic(results: GrangerAnalysisResults) -> CausalityMatrix:
	"""Compute the Granger F statistic per output and tested cause.

	Assumes the reference model is restricted relative to the base model.
	The restriction count may use structural or active-support degrees of
	freedom, according to ``results.dof_method``.
	"""
	values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)

	for cause_col, cause in enumerate(results.causes):
		base_rss = results.unrestricted_rss_per_output(cause)
		ref_rss = results.restricted_rss_per_output(cause)
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


@register_metric("wald_test", "wald", "wald_statistic")
def wald_test(results: GrangerAnalysisResults) -> CausalityMatrix:
    """Evaluate Wald statistics using the selected DOF policy.

    Args:
        results: Snapshots, preparation context, and DOF configuration.
    """
    method = results.dof_method

    if method not in {"structural", "active_support", "rank"}:
        raise ResultsError(
            f"Unsupported DOF method: {method!r}"
        )

    shape = (len(results.effects), len(results.causes))
    values = np.full(shape, np.nan, dtype=np.float64)
    q_matrix = np.zeros(shape, dtype=np.int_)
    status = np.full(shape, "no_tested_parameters", dtype=object)

    covariance_cache = {}

    for column, cause in enumerate(results.causes):
        unrestricted = results.get_snapshot(
            cause,
            "unrestricted",
        )

        u_mask, r_mask = results._effective_masks_for(cause)
        tested_mask = u_mask & ~r_mask

        q = results.n_restrictions_for(cause)
        q_matrix[:, column] = q

        if not np.any(q > 0):
            continue

        x, y, predictions = results._snapshot_coordinates(
            unrestricted
        )

        if method == "rank":
            statistics, reasons = rank_wald_test_value(
                y_true=y,
                y_pred=predictions,
                geometries=results._rank_geometry_for(cause),
            )

            values[:, column] = statistics

            for output, reason in enumerate(reasons):
                status[output, column] = (
                    "ok" if reason is None else reason
                )

                if q[output] > 0 and reason is not None:
                    warnings.warn(
                        f"Rank-mode Wald unavailable for "
                        f"cause={cause!r}, "
                        f"effect={results.effects[output]!r}: "
                        f"{reason}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            continue

        if unrestricted.weights is None:
            status[q > 0, column] = "missing_weights"
            continue

        weights = results._weights_as_feature_output(
            unrestricted.weights
        )

        if not np.isfinite(weights[u_mask]).all():
            raise ResultsError(
                f"Non-finite enabled coefficients for cause {cause!r}"
            )

        eligible = np.flatnonzero(tested_mask.any(axis=1))

        # Shared U snapshots can reuse covariance across causes.
        cache_key = (
            id(unrestricted),
            u_mask.shape,
            u_mask.tobytes(),
            tuple(eligible.tolist()),
        )

        if cache_key not in covariance_cache:
            covariance_cache[cache_key] = (
                estimate_masked_weight_covariance(
                    x_train=x,
                    y_true=y[:, eligible],
                    y_pred=predictions[:, eligible],
                    coefficient_mask=u_mask[eligible],
                    fit_intercept=unrestricted.bias_enabled,
                )
            )

        covariance = covariance_cache[cache_key]

        # Inactive stored values are not part of the effective model.
        effective_weights = np.where(
            u_mask[eligible],
            weights[eligible],
            0.0,
        )

        statistics = wald_test_value(
            weights=effective_weights,
            covariance=covariance,
            coefficient_mask=u_mask[eligible],
            start=0,
            end=u_mask.shape[1],
            tested_mask=tested_mask[eligible],
        )

        values[eligible, column] = statistics
        status[eligible, column] = "ok"

    hypothesis = (
        "nested_prediction_spaces"
        if method == "rank"
        else "zero_slope_coefficients"
    )

    return results.make_matrix(
        values,
        name="wald_test",
        metadata={
            "removed_signal_value": 0.0,
            "hypothesis": hypothesis,
            "dof_method": method,
            "active_support_tolerance": (
                results.active_support_tolerance
            ),
            "rank_rcond": results.rank_rcond,
            "n_restrictions": q_matrix.copy(),
            "status": status.copy(),
            "heuristic": method == "active_support",
        },
    )


@register_metric("likelihood_ratio", "likelihood_ratio_test", "lr", "lr_test")
def likelihood_ratio_test(results: GrangerAnalysisResults) -> CausalityMatrix:
	"""Compute the RSS-based likelihood-ratio statistic per output and cause.

	The reference and base predictions must use the same observations and
	output ordering. The statistic is evaluated against the base RSS.
	"""
	values = np.empty((len(results.effects), len(results.causes)), dtype=np.float64)
	for cause_col, cause in enumerate(results.causes):
		base_rss = results.unrestricted_rss_per_output(cause)
		ref_rss = results.restricted_rss_per_output(cause)
		values[:, cause_col] = likelihood_ratio_test_value(
			error_ref=ref_rss,
			error_base=base_rss,
			n_samples=results.n_samples,
		)
	return results.make_matrix(values, name="likelihood_ratio", metadata={"removed_signal_value": 0.0})

@register_metric(
    "f_statistic_p_value",
    "f_test_p_value",
    "f_p_value",
    "p_value_f",
)
def f_test_p_value(results: GrangerAnalysisResults) -> CausalityMatrix:
    """Compute F-test p-values using the current degrees-of-freedom policy.

    Args:
        results: Analysis results containing model snapshots and DOF settings.
    """
    statistic = results.evaluate(
        "f_statistic",
        restore_removed=False,
    )
    stat_values = statistic.data.loc[
        results.effects, results.causes
    ].to_numpy(dtype=np.float64)

    n_restrictions = np.empty(stat_values.shape, dtype=np.int_)
    residual_df = np.empty(stat_values.shape, dtype=np.int_)

    for column, cause in enumerate(results.causes):
        n_restrictions[:, column] = results.n_restrictions_for(cause)
        residual_df[:, column] = (
            results.n_samples
            - results.n_unrestricted_parameters(cause)
        )

    p_values = p_value_from_f_test(
        f_values=stat_values,
        n_restrictions=n_restrictions,
        residual_df=residual_df,
    )

    return results.make_matrix(
        p_values,
        name="f_statistic_p_value",
        metadata={
            "removed_signal_value": 1.0,
            "statistic_name": "f_statistic",
            "dof_method": results.dof_method,
            "active_support_tolerance": results.active_support_tolerance,
            "rank_rcond": results.rank_rcond,
            "n_restrictions": n_restrictions.copy(),
            "residual_df": residual_df.copy(),
        },
    )


@register_metric(
    "wald_test_p_value",
    "wald_statistic_p_value",
    "wald_p_value",
    "p_value_wald",
)
def wald_p_value(results: GrangerAnalysisResults) -> CausalityMatrix:
    """Convert Wald statistics using their recorded restriction counts."""
    statistic = results.evaluate(
        "wald_test",
        restore_removed=False,
    )

    stat_values = statistic.data.loc[
        results.effects,
        results.causes,
    ].to_numpy(dtype=np.float64)

    if "n_restrictions" not in statistic.metadata:
        raise ResultsError(
            "Wald statistic lacks restriction-count metadata"
        )

    q = np.asarray(
        statistic.metadata["n_restrictions"],
        dtype=np.int_,
    )

    if q.shape != stat_values.shape:
        raise ResultsError(
            "Wald restriction counts do not match the statistic shape"
        )

    p_values = p_value_from_wald_test(
        test_values=stat_values,
        df=q,
    )

    return results.make_matrix(
        p_values,
        name="wald_test_p_value",
        metadata={
            "removed_signal_value": 1.0,
            "statistic_name": "wald_test",
            "dof_method": statistic.metadata["dof_method"],
            "active_support_tolerance": statistic.metadata[
                "active_support_tolerance"
            ],
            "rank_rcond": statistic.metadata["rank_rcond"],
            "n_restrictions": q.copy(),
            "status": statistic.metadata["status"].copy(),
            "heuristic": statistic.metadata["heuristic"],
        },
    )


@register_metric(
    "likelihood_ratio_p_value",
    "likelihood_ratio_test_p_value",
    "lr_p_value",
    "p_value_lr",
    "p_value_likelihood_ratio",
)
def likelihood_ratio_p_value(results: GrangerAnalysisResults) -> CausalityMatrix:
    """Compute likelihood-ratio p-values using the current DOF policy.

    Args:
        results: Analysis results containing model snapshots and DOF settings.
    """
    statistic = results.evaluate(
        "likelihood_ratio",
        restore_removed=False,
    )
    stat_values = statistic.data.loc[
        results.effects, results.causes
    ].to_numpy(dtype=np.float64)

    n_restrictions = np.empty(stat_values.shape, dtype=np.int_)

    for column, cause in enumerate(results.causes):
        n_restrictions[:, column] = results.n_restrictions_for(cause)

    p_values = p_value_from_likelihood_ratio_test(
        test_values=stat_values,
        df=n_restrictions,
    )

    return results.make_matrix(
        p_values,
        name="likelihood_ratio_p_value",
        metadata={
            "removed_signal_value": 1.0,
            "statistic_name": "likelihood_ratio",
            "dof_method": results.dof_method,
            "active_support_tolerance": results.active_support_tolerance,
            "rank_rcond": results.rank_rcond,
            "n_restrictions": n_restrictions.copy(),
        },
    )

@register_metric("sign")
def sign_metric(results: GrangerAnalysisResults) -> CausalityMatrix:
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

def _rss_metric(
    results: GrangerAnalysisResults,
    *,
    role: str,
    normalize: bool,
) -> CausalityMatrix:
    """Build per-cause RSS or MSE for an explicit model role."""
    if normalize and results.n_samples <= 0:
        raise ResultsError(
            "MSE evaluation requires at least one observation"
        )

    values = np.empty(
        (len(results.effects), len(results.causes)),
        dtype=np.float64,
    )

    for column, cause in enumerate(results.causes):
        values[:, column] = results.rss_per_output(cause, role)

    if normalize:
        values /= float(results.n_samples)

    suffix = "mse" if normalize else "rss"

    return results.make_matrix(
        values,
        name=f"{role}_{suffix}",
        metadata={"model_role": role},
    )


@register_metric("unrestricted_rss")
def unrestricted_rss(results: Any) -> CausalityMatrix:
    """Return unrestricted RSS per effect and tested cause."""
    return _rss_metric(
        results,
        role="unrestricted",
        normalize=False,
    )


@register_metric("restricted_rss")
def restricted_rss(results: Any) -> CausalityMatrix:
    """Return restricted RSS per effect and tested cause."""
    return _rss_metric(
        results,
        role="restricted",
        normalize=False,
    )


@register_metric("unrestricted_mse")
def unrestricted_mse(results: Any) -> CausalityMatrix:
    """Return unrestricted MSE per effect and tested cause."""
    return _rss_metric(
        results,
        role="unrestricted",
        normalize=True,
    )


@register_metric("restricted_mse")
def restricted_mse(results: Any) -> CausalityMatrix:
    """Return restricted MSE per effect and tested cause."""
    return _rss_metric(
        results,
        role="restricted",
        normalize=True,
    )
