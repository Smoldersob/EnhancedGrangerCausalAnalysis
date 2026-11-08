"""Numerical statistics and distribution conversions.

This module owns statistic formulas and p-value conversions. Functions
accept plain arrays and return values together with per-item status
strings; they never access snapshots, result containers, or registries.
Callers in ``metrics.py`` orchestrate evaluation and record metadata.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, f

from ..core.exceptions import ResultsError


def ensure_2d(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a 2D float64 array, promoting 1D arrays to shape (n, 1)."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ResultsError(f"Expected 1D or 2D array, got shape {arr.shape}")
    return arr


def residual_sum_of_squares(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute RSS per output: sum_t (y_true - y_pred)^2."""
    true_2d = ensure_2d(y_true)
    pred_2d = ensure_2d(y_pred)
    if true_2d.shape != pred_2d.shape:
        raise ResultsError(
            f"y_true and y_pred must have the same shape, got {true_2d.shape} vs {pred_2d.shape}"
        )
    return np.sum((true_2d - pred_2d) ** 2, axis=0)


def f_statistic_values(
    rss_u,
    rss_r,
    restriction_df,
    residual_df,
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute nested-model F values from RSS and float DOF.

    Degrees of freedom are not coerced to integers. When the restricted
    model attains a lower RSS than the unrestricted model, the value is
    clipped to zero and the status reports it instead of hiding it.
    """
    u = np.atleast_1d(np.asarray(rss_u, dtype=float))
    r = np.atleast_1d(np.asarray(rss_r, dtype=float))
    q = np.atleast_1d(np.asarray(restriction_df, dtype=float))
    df = np.atleast_1d(np.asarray(residual_df, dtype=float))

    if not (u.shape == r.shape == q.shape == df.shape):
        raise ResultsError("F inputs must share a single shape")

    values = np.full(u.shape, np.nan)
    statuses: list[str] = []

    for index in range(u.size):
        if not (
            np.isfinite(u[index])
            and np.isfinite(r[index])
            and u[index] >= 0
            and r[index] >= 0
        ):
            statuses.append("invalid_rss")
            continue

        if not np.isfinite(q[index]) or q[index] <= 0:
            statuses.append("invalid_restriction_df")
            continue

        if not np.isfinite(df[index]) or df[index] <= 0:
            statuses.append("invalid_residual_df")
            continue

        if u[index] == 0 and r[index] == 0:
            statuses.append("zero_rss_both_models")
            continue

        if r[index] < u[index]:
            values[index] = 0.0
            statuses.append("restricted_rss_below_unrestricted")
            continue

        if u[index] == 0:
            values[index] = np.inf
            statuses.append("ok")
            continue

        values[index] = ((r[index] - u[index]) / q[index]) / (
            u[index] / df[index]
        )
        statuses.append("ok")

    return values, statuses


def likelihood_ratio_values(
    rss_u,
    rss_r,
    n_samples: int,
) -> tuple[NDArray[np.float64], list[str]]:
    """Compute RSS log-ratio values without epsilon clipping."""
    u = np.atleast_1d(np.asarray(rss_u, dtype=float))
    r = np.atleast_1d(np.asarray(rss_r, dtype=float))

    if u.shape != r.shape:
        raise ResultsError("RSS inputs must share a single shape")

    values = np.full(u.shape, np.nan)
    statuses: list[str] = []

    for index in range(u.size):
        if n_samples <= 0:
            statuses.append("no_observations")
            continue

        if not (
            np.isfinite(u[index])
            and np.isfinite(r[index])
            and u[index] >= 0
            and r[index] >= 0
        ):
            statuses.append("invalid_rss")
            continue

        if u[index] == 0 and r[index] == 0:
            statuses.append("zero_rss_both_models")
            continue

        if r[index] < u[index]:
            values[index] = 0.0
            statuses.append("restricted_rss_below_unrestricted")
            continue

        if u[index] == 0:
            values[index] = np.inf
            statuses.append("ok")
            continue

        values[index] = float(n_samples) * (
            np.log(r[index]) - np.log(u[index])
        )
        statuses.append("ok")

    return values, statuses


def f_test_p_values(
    statistics,
    restriction_df,
    residual_df,
) -> tuple[NDArray[np.float64], list[str]]:
    """Convert F values to upper-tail p-values using float DOF."""
    values = np.atleast_1d(np.asarray(statistics, dtype=float))
    q = np.atleast_1d(np.asarray(restriction_df, dtype=float))
    df = np.atleast_1d(np.asarray(residual_df, dtype=float))

    if not (values.shape == q.shape == df.shape):
        raise ResultsError("F p-value inputs must share a single shape")

    p_values = np.full(values.shape, np.nan)
    statuses: list[str] = []

    for index in range(values.size):
        value = values[index]

        if np.isnan(value) or value < 0:
            statuses.append("statistic_unavailable")
            continue

        if not np.isfinite(q[index]) or q[index] <= 0:
            statuses.append("invalid_restriction_df")
            continue

        if not np.isfinite(df[index]) or df[index] <= 0:
            statuses.append("invalid_residual_df")
            continue

        p_value = f.sf(
            value, dfn=q[index], dfd=df[index]
        )
        p_values[index] = p_value
        statuses.append(
            "ok"
            if np.isfinite(p_value)
            else "distribution_evaluation_failed"
        )

    return p_values, statuses


def chi_square_p_values(
    statistics,
    restriction_df,
) -> tuple[NDArray[np.float64], list[str]]:
    """Convert chi-square values to upper-tail p-values using float DOF."""
    values = np.atleast_1d(np.asarray(statistics, dtype=float))
    q = np.atleast_1d(np.asarray(restriction_df, dtype=float))

    if values.shape != q.shape:
        raise ResultsError("Chi-square p-value inputs must share a single shape")

    p_values = np.full(values.shape, np.nan)
    statuses: list[str] = []

    for index in range(values.size):
        value = values[index]

        if np.isnan(value) or value < 0:
            statuses.append("statistic_unavailable")
            continue

        if not np.isfinite(q[index]) or q[index] <= 0:
            statuses.append("invalid_restriction_df")
            continue

        p_value = chi2.sf(value, df=q[index])
        p_values[index] = p_value
        statuses.append(
            "ok"
            if np.isfinite(p_value)
            else "distribution_evaluation_failed"
        )

    return p_values, statuses


def ols_projection_error(
    basis,
    target,
    prediction,
) -> tuple[float, float]:
    """Return (error, tolerance) for a prediction against its OLS projection.

    ``basis`` is an orthonormal basis of a retained design space. The
    error compares the stored prediction with the OLS projection of the
    target onto that space, so the check covers both membership in the
    space and equivalence with the OLS fit.
    """
    basis = np.asarray(basis, dtype=float)
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    expected = basis @ (basis.T @ target)
    error = float(np.linalg.norm(prediction - expected))
    tolerance = 1e-7 * max(1.0, float(np.linalg.norm(target)))
    return error, tolerance


def nested_design_geometry(design_u, design_r, rcond=None):
    """Compute design ranks and optional nested-space geometry.

    Rank counts remain available when independent SVD truncations are
    not nested. In that case, basis_added is None and must not be used
    by a projection-based Wald calculation.
    """
    u = np.asarray(design_u, dtype=float)
    r = np.asarray(design_r, dtype=float)

    if u.ndim != 2 or r.ndim != 2 or u.shape[0] != r.shape[0]:
        raise ResultsError("Nested designs must be 2D with matching rows")

    if not np.isfinite(u).all() or not np.isfinite(r).all():
        raise ResultsError("Designs must contain finite values")

    if u.shape[0] == 0:
        raise ResultsError("Designs must contain observations")

    if rcond is not None and (
        isinstance(rcond, (bool, np.bool_))
        or not np.isfinite(rcond)
        or not 0 <= rcond < 1
    ):
        raise ResultsError(
            "rank_rcond must be None or a finite number in [0, 1)"
        )

    rel = (
        np.finfo(float).eps * max(u.shape)
        if rcond is None
        else float(rcond)
    )

    qu_full, su, _ = np.linalg.svd(u, full_matrices=False)
    qr_full, sr, _ = np.linalg.svd(r, full_matrices=False)

    cutoff = rel * (float(su[0]) if su.size else 0.0)

    ru = int(np.count_nonzero(su > cutoff))
    rr = int(np.count_nonzero(sr > cutoff))

    qu = qu_full[:, :ru]
    qr = qr_full[:, :rr]

    if rr > ru:
        raise ResultsError(
            "Restricted numerical rank exceeds unrestricted rank"
        )

    q = ru - rr

    nesting_error = float(
        np.linalg.norm(
            qr - qu @ (qu.T @ qr),
            ord="fro",
        )
    )
    nesting_tolerance = 1e-7 * max(1.0, np.sqrt(rr))
    is_nested = nesting_error <= nesting_tolerance

    qa = None
    added_basis_valid = False
    geometry_status = "non_nested_after_truncation"

    if is_nested:
        if q == 0:
            qa = np.empty((u.shape[0], 0))
            added_basis_valid = True
            geometry_status = "nested"
        else:
            increment = qu - qr @ (qr.T @ qu)
            qa_full, sa, _ = np.linalg.svd(
                increment,
                full_matrices=False,
            )

            if int(np.count_nonzero(sa > 0.5)) == q:
                qa = qa_full[:, :q]
                added_basis_valid = True
                geometry_status = "nested"
            else:
                geometry_status = "unstable_incremental_basis"

    return dict(
        rank_u=ru,
        rank_r=rr,
        q=q,
        cutoff=cutoff,
        singular_u=su,
        singular_r=sr,
        basis_u=qu,
        basis_r=qr,
        basis_added=qa,
        nesting_error=nesting_error,
        nesting_tolerance=nesting_tolerance,
        is_nested=is_nested,
        added_basis_valid=added_basis_valid,
        geometry_status=geometry_status,
    )


def rank_wald_test_value(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    geometries: list[dict],
) -> tuple[NDArray[np.float64], list[str | None]]:
    """Compute projection-based Wald statistics per output.

    Args:
        y_true: Targets in the unrestricted model coordinates.
        y_pred: Existing unrestricted predictions in the same coordinates.
        geometries: Nested-design geometry for each output.

    Returns:
        Statistics and optional reasons for unavailable outputs.
    """
    y = ensure_2d(y_true)
    predictions = ensure_2d(y_pred)

    if y.shape != predictions.shape:
        raise ResultsError(
            "Targets and predictions must have matching shapes"
        )

    if not np.isfinite(y).all() or not np.isfinite(predictions).all():
        raise ResultsError(
            "Rank-Wald targets and predictions must be finite"
        )

    n_samples, n_outputs = y.shape

    if len(geometries) != n_outputs:
        raise ResultsError(
            "One rank geometry is required for each output"
        )

    values = np.full(n_outputs, np.nan, dtype=np.float64)
    reasons: list[str | None] = [None] * n_outputs

    for output, geometry in enumerate(geometries):
        q = int(geometry["q"])

        if q == 0:
            reasons[output] = "no additional retained directions"
            continue

        if not geometry["added_basis_valid"]:
            reasons[output] = (
                "incremental prediction space is unavailable: "
                f"{geometry['geometry_status']}"
            )
            continue

        residual_df = n_samples - int(geometry["rank_u"])

        if residual_df <= 0:
            reasons[output] = (
                "residual degrees of freedom are not positive"
            )
            continue

        basis_u = np.asarray(
            geometry["basis_u"],
            dtype=np.float64,
        )
        basis_added = np.asarray(
            geometry["basis_added"],
            dtype=np.float64,
        )

        if (
            basis_u.shape != (n_samples, int(geometry["rank_u"]))
            or basis_added.shape != (n_samples, q)
        ):
            raise ResultsError("Invalid rank-Wald basis dimensions")

        target = y[:, output]
        prediction = predictions[:, output]

        retained_prediction = basis_u @ (
            basis_u.T @ prediction
        )

        projection_error = np.linalg.norm(
            prediction - retained_prediction
        )
        projection_tolerance = 1e-7 * max(
            1.0,
            np.linalg.norm(prediction),
        )

        if projection_error > projection_tolerance:
            reasons[output] = (
                "predictions are outside the retained rank space "
                f"(error={projection_error:.6g}, "
                f"tolerance={projection_tolerance:.6g})"
            )
            continue

        residual = target - prediction

        score = np.linalg.norm(basis_u.T @ residual)
        score_tolerance = 1e-7 * max(
            1.0,
            np.linalg.norm(basis_u.T @ target),
        )

        if score > score_tolerance:
            reasons[output] = (
                "the unrestricted fit is not OLS-equivalent "
                "in the retained rank space "
                f"(error={score:.6g}, "
                f"tolerance={score_tolerance:.6g})"
            )
            continue

        sigma2 = float(
            np.sum(residual**2) / residual_df
        )
        numerator = float(
            np.sum((basis_added.T @ prediction) ** 2)
        )

        if not np.isfinite(sigma2) or not np.isfinite(numerator):
            reasons[output] = (
                "residual variance or numerator is non-finite"
            )
            continue

        if sigma2 > 0:
            values[output] = numerator / sigma2
        elif numerator > 0:
            values[output] = np.inf
        else:
            reasons[output] = (
                "residual variance and numerator are both zero"
            )

    return values, reasons


def coefficient_wald_value(
    design,
    coefficients,
    tested_positions,
    residual,
    residual_df,
    *,
    rcond=None,
):
    """Compute a coefficient Wald statistic for one output.

    Inputs use one shared coefficient space. The design must have
    full retained column rank. Returns a value and availability status.
    """
    design = np.asarray(design, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    tested = np.asarray(tested_positions, dtype=int)
    residual = np.asarray(residual, dtype=float)

    if (
        design.ndim != 2
        or coefficients.shape != (design.shape[1],)
        or residual.shape != (design.shape[0],)
    ):
        raise ResultsError("Invalid coefficient-Wald dimensions")

    if tested.size == 0:
        return np.nan, "no_tested_parameters"

    if not (
        np.isfinite(design).all()
        and np.isfinite(coefficients).all()
        and np.isfinite(residual).all()
    ):
        return np.nan, "nonfinite_wald_inputs"

    if not np.isfinite(residual_df) or residual_df <= 0:
        return np.nan, "invalid_residual_df"

    _, singular, vt = np.linalg.svd(design, full_matrices=False)

    relative = (
        np.finfo(float).eps * max(design.shape)
        if rcond is None
        else float(rcond)
    )
    cutoff = relative * singular[0] if singular.size else 0.0

    if np.count_nonzero(singular > cutoff) != design.shape[1]:
        return np.nan, "coefficient_design_rank_deficient"

    sigma2 = float(residual @ residual / residual_df)

    # Rows of this factor correspond to coefficient coordinates.
    factor = vt.T / singular
    tested_factor = factor[tested]
    covariance_unit = tested_factor @ tested_factor.T
    covariance_unit = 0.5 * (
        covariance_unit + covariance_unit.T
    )
    beta = coefficients[tested]

    if sigma2 == 0:
        if np.any(beta != 0):
            return np.inf, "zero_residual_variance"
        return np.nan, "zero_variance_and_zero_effect"

    try:
        np.linalg.cholesky(covariance_unit)
        value = float(
            beta @ np.linalg.solve(covariance_unit, beta)
            / sigma2
        )
    except np.linalg.LinAlgError:
        return np.nan, "singular_tested_covariance"

    if np.isnan(value) or value < 0:
        return np.nan, "invalid_wald_value"

    return value, "ok"
