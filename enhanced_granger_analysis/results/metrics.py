"""Registered causality metrics.

Metrics orchestrate per-cause evaluation over stored snapshots and
delegate numerical computation to ``statistics.py``. This module does
not implement statistic formulas.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

from ..core.exceptions import RegistryError, ResultsError
from ..core.registry import Registry
from .causality_matrix import CausalityMatrix
from .dof import dof_name
from .granger_results import GrangerAnalysisResults
from .statistics import (
    chi_square_p_values,
    coefficient_wald_value,
    f_statistic_values,
    f_test_p_values,
    likelihood_ratio_values,
    ols_projection_error,
    rank_wald_test_value,
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


def _empty_test(results: GrangerAnalysisResults):
    """Allocate values and diagnostics in effective matrix order."""
    shape = (len(results.effects), len(results.causes))
    diagnostics = np.empty(shape, dtype=object)

    for index in np.ndindex(shape):
        diagnostics[index] = []

    return np.full(shape, np.nan), {
        "removed_signal_value": 0.0,
        "dof_method": dof_name(results.dof_method),
        "dof_params": dict(results.dof_params),
        "n_restrictions": np.full(shape, np.nan),
        "residual_df": np.full(shape, np.nan),
        "status": np.full(shape, "not_evaluated", dtype=object),
        "dof_status": np.full(shape, "", dtype=object),
        "diagnostics": diagnostics,
        "heuristic": np.zeros(shape, dtype=bool),
        "calibration": "not_verified",
        "effective_effects": tuple(results.effects),
        "effective_causes": tuple(results.causes),
        "diagnostic_scope": "effective",
    }


def _record_dof(metadata, column, dimensions):
    """Record the exact DOF used by a statistic."""
    metadata["n_restrictions"][:, column] = (
        dimensions.restriction_df
    )
    metadata["residual_df"][:, column] = (
        dimensions.residual_df
    )
    metadata["dof_status"][:, column] = dimensions.status
    metadata["heuristic"][:, column] = dimensions.heuristic

    for output, status in enumerate(dimensions.status):
        if status != "ok":
            metadata["diagnostics"][output, column].append(
                f"dof:{status}"
            )


def _comparison_diagnostics(results, cause):
    """Check linear geometry without blocking RSS-based measures.

    Missing coordinate data is reported as unavailable diagnostics.
    Nonlinear or regularized predictions may legitimately fail OLS checks.
    """
    count = len(results.effects)
    issues = [[] for _ in range(count)]

    try:
        geometries = results._rank_geometry_for(cause)
    except (ValueError, np.linalg.LinAlgError) as exc:
        for row in issues:
            row.append(
                f"linear_geometry_unavailable:{type(exc).__name__}"
            )
        return issues

    for output, geometry in enumerate(geometries):
        if not geometry["is_nested"]:
            issues[output].append(
                "linear_spaces_non_nested_after_truncation"
            )
        if not geometry["added_basis_valid"]:
            issues[output].append(
                "incremental_linear_basis_unavailable"
            )

    for role, basis_key in (
        ("unrestricted", "basis_u"),
        ("restricted", "basis_r"),
    ):
        snapshot = results.get_snapshot(cause, role)

        try:
            _, target, prediction = results._snapshot_coordinates(
                snapshot
            )
        except ValueError:
            for row in issues:
                row.append(f"{role}_ols_check_unavailable")
            continue

        for output, geometry in enumerate(geometries):
            error, tolerance = ols_projection_error(
                geometry[basis_key],
                target[:, output],
                prediction[:, output],
            )

            if error > tolerance:
                issues[output].append(
                    f"{role}_not_ols_in_retained_linear_space"
                )

    return issues


def _attach_comparison_diagnostics(
    results,
    cause,
    column,
    metadata,
    *,
    enabled,
):
    """Attach optional assumptions diagnostics."""
    if not enabled:
        return

    issues = _comparison_diagnostics(results, cause)

    for output, messages in enumerate(issues):
        metadata["diagnostics"][output, column].extend(messages)
        if messages:
            metadata["heuristic"][output, column] = True


def _record_clipped_status(metadata, output, column, statuses):
    """Record statuses and flag numerically clipped comparisons."""
    for index, status in enumerate(statuses):
        metadata["status"][output + index, column] = status

        if status == "restricted_rss_below_unrestricted":
            metadata["diagnostics"][output + index, column].append(
                status
            )
            metadata["heuristic"][output + index, column] = True


@register_metric("f_statistic", "f")
def f_statistic(
    results: GrangerAnalysisResults,
    *,
    diagnose: bool = True,
) -> CausalityMatrix:
    """Compute RSS-based F values using the selected DOF strategy."""
    values, metadata = _empty_test(results)
    metadata["reference_distribution"] = "F"

    for column, cause in enumerate(results.causes):
        dimensions = results.dof_for(cause)
        _record_dof(metadata, column, dimensions)

        _attach_comparison_diagnostics(
            results, cause, column, metadata, enabled=diagnose
        )

        column_values, statuses = f_statistic_values(
            results.rss_per_output(cause, "unrestricted"),
            results.rss_per_output(cause, "restricted"),
            dimensions.restriction_df,
            dimensions.residual_df,
        )
        values[:, column] = column_values
        _record_clipped_status(metadata, 0, column, statuses)

    return results.make_matrix(
        values, name="f_statistic", metadata=metadata
    )


@register_metric(
    "likelihood_ratio",
    "likelihood_ratio_test",
    "lr",
    "lr_test",
)
def likelihood_ratio_test(
    results: GrangerAnalysisResults,
    *,
    diagnose: bool = True,
) -> CausalityMatrix:
    """Compute RSS log-ratio values without epsilon clipping."""
    values, metadata = _empty_test(results)
    metadata["reference_distribution"] = "chi2"
    metadata["interpretation"] = "rss_log_ratio"

    for column, cause in enumerate(results.causes):
        dimensions = results.dof_for(cause)
        _record_dof(metadata, column, dimensions)

        _attach_comparison_diagnostics(
            results, cause, column, metadata, enabled=diagnose
        )

        column_values, statuses = likelihood_ratio_values(
            results.rss_per_output(cause, "unrestricted"),
            results.rss_per_output(cause, "restricted"),
            results.n_samples,
        )
        values[:, column] = column_values
        _record_clipped_status(metadata, 0, column, statuses)

    return results.make_matrix(
        values, name="likelihood_ratio", metadata=metadata
    )


@register_metric("wald", "wald_test", "wald_statistic")
def wald_test(
    results: GrangerAnalysisResults,
    *,
    diagnose: bool = True,
) -> CausalityMatrix:
    """Compute coefficient Wald statistics using U weights."""
    values, metadata = _empty_test(results)
    metadata["hypothesis"] = "zero_slope_coefficients"
    metadata["reference_distribution"] = "chi2"

    for column, cause in enumerate(results.causes):
        dimensions = results.dof_for(cause)
        _record_dof(metadata, column, dimensions)

        u_mask, r_mask = results._structural_masks_for(cause)
        tested_mask = u_mask & ~r_mask
        snapshot = results.get_snapshot(cause, "unrestricted")

        _attach_comparison_diagnostics(
            results, cause, column, metadata, enabled=diagnose
        )

        if snapshot.weights is None:
            metadata["status"][:, column] = "missing_weights"
            continue

        weights = results._weights_as_feature_output(
            snapshot.weights
        )

        try:
            x, y, prediction = results._snapshot_coordinates(
                snapshot
            )
        except ValueError as exc:
            metadata["status"][:, column] = (
                "snapshot_coordinates_unavailable"
            )
            for output in range(len(results.effects)):
                metadata["diagnostics"][output, column].append(
                    str(exc)
                )
            continue

        for output in range(len(results.effects)):
            active = np.flatnonzero(u_mask[output])
            local_tested = np.flatnonzero(
                tested_mask[output, active]
            )

            if local_tested.size == 0:
                metadata["status"][output, column] = (
                    "no_tested_parameters"
                )
                continue

            design = x[:, active]
            coefficients = weights[output, active]

            if snapshot.bias_enabled:
                design = np.column_stack((
                    np.ones(len(x)),
                    design,
                ))
                # The intercept is not tested.
                coefficients = np.r_[0.0, coefficients]
                local_tested = local_tested + 1

            q = dimensions.restriction_df[output]

            if q != local_tested.size:
                metadata["diagnostics"][output, column].append(
                    "selected_df_differs_from_coefficient_count"
                )
                metadata["heuristic"][output, column] = True

            value, status = coefficient_wald_value(
                design=design,
                coefficients=coefficients,
                tested_positions=local_tested,
                residual=y[:, output] - prediction[:, output],
                residual_df=dimensions.residual_df[output],
                rcond=results.dof_params.get("rank_rcond"),
            )

            values[output, column] = value
            metadata["status"][output, column] = status

    return results.make_matrix(
        values, name="wald", metadata=metadata
    )


@register_metric(
    "wald_projection",
    "wald_svd",
    "wald_projection_test",
)
def wald_projection(
    results: GrangerAnalysisResults,
    *,
    diagnose: bool = True,
) -> CausalityMatrix:
    """Compute projection Wald statistics in retained linear spaces."""
    values, metadata = _empty_test(results)
    metadata["hypothesis"] = "nested_prediction_spaces"
    metadata["reference_distribution"] = "chi2"

    for column, cause in enumerate(results.causes):
        dimensions = results.dof_for(cause)
        _record_dof(metadata, column, dimensions)

        try:
            geometries = results._rank_geometry_for(cause)
            snapshot = results.get_snapshot(
                cause, "unrestricted"
            )
            _, y, prediction = results._snapshot_coordinates(
                snapshot
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            metadata["status"][:, column] = (
                "projection_inputs_unavailable"
            )
            for output in range(len(results.effects)):
                metadata["diagnostics"][output, column].append(
                    str(exc)
                )
            continue

        _attach_comparison_diagnostics(
            results, cause, column, metadata, enabled=diagnose
        )

        statistics, reasons = rank_wald_test_value(
            y_true=y,
            y_pred=prediction,
            geometries=geometries,
        )

        for output, geometry in enumerate(geometries):
            expected_q = float(geometry["q"])
            expected_df = float(
                results.n_samples - geometry["rank_u"]
            )

            q = dimensions.restriction_df[output]
            df = dimensions.residual_df[output]

            if not (
                np.isfinite(q)
                and np.isfinite(df)
                and np.isclose(q, expected_q, rtol=1e-10, atol=1e-12)
                and np.isclose(
                    df, expected_df, rtol=1e-10, atol=1e-12
                )
            ):
                metadata["status"][output, column] = (
                    "dof_incompatible_with_projection"
                )
                continue

            values[output, column] = statistics[output]
            metadata["status"][output, column] = (
                reasons[output]
                if reasons[output] is not None
                else "ok"
            )

    return results.make_matrix(
        values, name="wald_projection", metadata=metadata
    )


def _test_p_value(
    results,
    statistic_name,
    output_name,
    *,
    distribution,
    diagnose=True,
):
    """Convert a statistic using its recorded DOF and diagnostics."""
    statistic = results.evaluate(
        statistic_name,
        restore_removed=False,
        diagnose=diagnose,
    )
    values = statistic.data.to_numpy(dtype=float)

    metadata = deepcopy(statistic.metadata)
    q = np.asarray(metadata["n_restrictions"], dtype=float)
    df = np.asarray(metadata["residual_df"], dtype=float)

    if distribution == "F":
        p_flat, statuses = f_test_p_values(
            values.ravel(), q.ravel(), df.ravel()
        )
    else:
        p_flat, statuses = chi_square_p_values(
            values.ravel(), q.ravel()
        )

    p_values = np.asarray(p_flat).reshape(values.shape)
    p_status = np.array(statuses, dtype=object).reshape(values.shape)

    metadata["statistic_name"] = statistic.name
    metadata["statistic_status"] = metadata["status"].copy()
    metadata["status"] = p_status
    metadata["removed_signal_value"] = 1.0

    return results.make_matrix(
        p_values, name=output_name, metadata=metadata
    )


@register_metric(
    "f_statistic_p_value",
    "f_test_p_value",
    "f_p_value",
    "p_value_f",
)
def f_test_p_value(results, *, diagnose=True):
    """Convert F values to reference-distribution p-values."""
    return _test_p_value(
        results,
        "f_statistic",
        "f_statistic_p_value",
        distribution="F",
        diagnose=diagnose,
    )


@register_metric(
    "wald_p_value",
    "wald_test_p_value",
    "wald_statistic_p_value",
    "p_value_wald",
)
def wald_p_value(results, *, diagnose=True):
    """Convert coefficient Wald statistics to chi-square p-values."""
    return _test_p_value(
        results,
        "wald",
        "wald_p_value",
        distribution="chi2",
        diagnose=diagnose,
    )


@register_metric(
    "wald_projection_p_value",
    "wald_svd_p_value",
    "wald_projection_test_p_value",
    "p_value_wald_projection",
)
def wald_projection_p_value(results, *, diagnose=True):
    """Convert projection Wald statistics to chi-square p-values."""
    return _test_p_value(
        results,
        "wald_projection",
        "wald_projection_p_value",
        distribution="chi2",
        diagnose=diagnose,
    )


@register_metric(
    "likelihood_ratio_p_value",
    "likelihood_ratio_test_p_value",
    "lr_p_value",
    "p_value_lr",
    "p_value_likelihood_ratio",
)
def likelihood_ratio_p_value(results, *, diagnose=True):
    """Convert RSS log-ratio values to chi-square p-values."""
    return _test_p_value(
        results,
        "likelihood_ratio",
        "likelihood_ratio_p_value",
        distribution="chi2",
        diagnose=diagnose,
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
        snapshot = results.get_snapshot(cause, "unrestricted")
        if snapshot.weights is None or snapshot.active_slope_mask is None:
            continue
        weights = results._weights_as_feature_output(snapshot.weights)
        if weights.shape[0] != len(results.effects):
            raise ValueError(
                f"Weight shape {weights.shape} is incompatible with "
                f"{len(results.effects)} effects"
            )
        restricted = results.get_snapshot(cause, "restricted")
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
