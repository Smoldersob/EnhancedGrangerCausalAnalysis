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


def estimate_masked_weight_covariance(
	x_train: NDArray[np.float64],
	y_true: NDArray[np.float64],
	y_pred: NDArray[np.float64],
	coefficient_mask: NDArray[np.float64],
	fit_intercept: bool = True,
) -> NDArray[np.float64]:
	"""
	Estimate OLS covariance matrices for a fixed masked VAR/VARX model.

	The returned array has shape:
		(n_outputs, n_features, n_features)

	Its rows/columns corresponding to inactive mask entries are zero.
	"""
	x_arr = np.asarray(x_train, dtype=np.float64)
	y_true_2d = ensure_2d(y_true)
	y_pred_2d = ensure_2d(y_pred)
	mask_arr = np.asarray(coefficient_mask, dtype=bool)

	if x_arr.ndim != 2:
		raise ResultsError(f"x_train must be 2D, got {x_arr.shape}")

	if y_true_2d.shape != y_pred_2d.shape:
		raise ResultsError(
			f"y_true and y_pred must match, got "
			f"{y_true_2d.shape} vs {y_pred_2d.shape}"
		)

	n_samples, n_features = x_arr.shape
	n_outputs = y_true_2d.shape[1]

	if y_true_2d.shape[0] != n_samples:
		raise ResultsError(
			f"x_train and y_true must have equal sample count, got "
			f"{n_samples} vs {y_true_2d.shape[0]}"
		)

	if mask_arr.shape != (n_outputs, n_features):
		raise ResultsError(
			f"coefficient_mask must have shape {(n_outputs, n_features)}, "
			f"got {mask_arr.shape}"
		)

	residuals = y_true_2d - y_pred_2d
	covariance = np.zeros((n_outputs, n_features, n_features), dtype=np.float64)

	for out_idx in range(n_outputs):
		active_idx = np.flatnonzero(mask_arr[out_idx])

		if active_idx.size == 0:
			continue

		x_active = x_arr[:, active_idx]

		if fit_intercept:
			design = np.column_stack([np.ones(n_samples, dtype=np.float64), x_active])
		else:
			design = x_active

		n_parameters = design.shape[1]
		residual_df = n_samples - n_parameters

		if residual_df <= 0:
			raise ResultsError(
				f"Output {out_idx}: insufficient residual degrees of freedom; "
				f"n_samples={n_samples}, n_parameters={n_parameters}"
			)

		design_rank = np.linalg.matrix_rank(design)
		if design_rank != n_parameters:
			raise ResultsError(
				f"Output {out_idx}: active design is rank-deficient; "
				f"rank={design_rank}, n_parameters={n_parameters}"
			)

		sigma2 = np.sum(residuals[:, out_idx] ** 2) / float(residual_df)
		covariance_active = sigma2 * np.linalg.inv(design.T @ design)

		if fit_intercept:
			covariance_active = covariance_active[1:, 1:]

		covariance[out_idx][np.ix_(active_idx, active_idx)] = covariance_active

	return covariance


def error_values(
	y_true: NDArray[np.float64],
	y_base_pred: NDArray[np.float64],
	y_ref_pred: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
	"""Compute normalized base and reference errors per output."""
	y_true_2d = ensure_2d(y_true)
	rss_base = residual_sum_of_squares(y_true_2d, y_base_pred)
	rss_ref = residual_sum_of_squares(y_true_2d, y_ref_pred)

	n_samples = y_true_2d.shape[0]
	base_error = rss_base / float(n_samples)
	ref_error = rss_ref / float(n_samples)
	return base_error, ref_error


def wald_test_value(
	weights: NDArray[np.float64],
	covariance: NDArray[np.float64],
	coefficient_mask: NDArray[np.float64],
	start: int,
	end: int,
	tested_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
	"""Return one Wald statistic per output; NaN if structurally untestable."""
	weights_arr = np.asarray(weights, dtype=np.float64)
	cov_arr = np.asarray(covariance, dtype=np.float64)
	mask_arr = np.asarray(coefficient_mask, dtype=bool)

	n_outputs, n_features = weights_arr.shape

	if mask_arr.shape != weights_arr.shape:
		raise ResultsError(
			f"Mask shape {mask_arr.shape} does not match weights shape {weights_arr.shape}"
		)

	if cov_arr.shape != (n_outputs, n_features, n_features):
		raise ResultsError(
			"Invalid covariance shape: "
			f"expected {(n_outputs, n_features, n_features)}, got {cov_arr.shape}"
		)

	if not (0 <= start < end <= n_features):
		raise ResultsError(f"Invalid tested coefficient block [{start}:{end}]")

	if tested_mask is not None:
		tested_arr = np.asarray(tested_mask, dtype=bool)
		if tested_arr.shape != mask_arr.shape:
			raise ResultsError(
				f"tested_mask shape {tested_arr.shape} does not match {mask_arr.shape}"
			)
	else:
		tested_arr = np.zeros_like(mask_arr, dtype=bool)
		tested_arr[:, start:end] = True
	statistics = np.full(n_outputs, np.nan, dtype=np.float64)

	for out_idx in range(n_outputs):
		tested_idx = np.flatnonzero(mask_arr[out_idx] & tested_arr[out_idx])

		if tested_idx.size == 0:
			continue

		beta_block = weights_arr[out_idx, tested_idx]
		cov_block = cov_arr[out_idx][np.ix_(tested_idx, tested_idx)]
		cov_block = 0.5 * (cov_block + cov_block.T)

		if np.linalg.matrix_rank(cov_block) != tested_idx.size:
			raise ResultsError(
				f"Singular Wald covariance for output {out_idx}, tested indices={tested_idx.tolist()}"
			)

		statistics[out_idx] = float(beta_block @ np.linalg.solve(cov_block, beta_block))

	return statistics


def f_test_value(error_ref, error_base, n_restrictions,
				 n_unrestricted_parameters, n_samples):
	"""Compute nested-model F statistics independently for each output."""
	r = np.asarray(error_ref, dtype=float)
	u = np.asarray(error_base, dtype=float)
	if r.shape != u.shape:
		raise ResultsError("Restricted and unrestricted RSS shapes differ")
	q = np.broadcast_to(np.asarray(n_restrictions, dtype=float), u.shape)
	k = np.broadcast_to(np.asarray(n_unrestricted_parameters, dtype=float), u.shape)
	if (not np.isfinite(r).all() or not np.isfinite(u).all()
			or np.any(r < 0) or np.any(u < 0)):
		raise ResultsError("RSS must be finite and nonnegative")
	if (not np.isfinite(q).all() or not np.isfinite(k).all()
			or np.any(q < 0) or np.any(k < 0)
			or np.any(q != np.floor(q)) or np.any(k != np.floor(k))):
		raise ResultsError("Parameter and restriction counts must be nonnegative integers")
	df = n_samples - k
	if np.any(df <= 0):
		raise ResultsError("Unrestricted residual degrees of freedom must be positive")
	valid = q > 0
	improvement = np.maximum(r - u, 0.0)
	values = np.full(u.shape, np.nan)
	positive = valid & (u > 0)
	values[positive] = ((np.maximum(improvement[positive], 0) / q[positive])
						/ (u[positive] / df[positive]))
	values[valid & (u == 0) & (r > 0)] = np.inf
	return values


def likelihood_ratio_test_value(
	error_ref: NDArray[np.float64],
	error_base: NDArray[np.float64],
	n_samples: int,
) -> NDArray[np.float64]:
	"""Compute likelihood-ratio statistic vector from reference and base errors."""
	err_ref = np.asarray(error_ref, dtype=np.float64)
	err_base = np.asarray(error_base, dtype=np.float64)
	if err_ref.shape != err_base.shape:
		raise ResultsError("error_ref and error_base must have the same shape")

	ratio = np.maximum(err_ref, np.finfo(np.float64).eps) / np.maximum(
		err_base, np.finfo(np.float64).eps
	)
	return float(n_samples) * np.maximum(np.log(ratio), 0.0)


def p_value_from_f_test(
	f_values: NDArray[np.float64],
	n_restrictions: NDArray[np.int_],
	residual_df: NDArray[np.int_],
) -> NDArray[np.float64]:
	"""Convert per-output F statistics to upper-tail p-values."""
	f_values_arr = np.asarray(f_values, dtype=np.float64)
	n_restrictions_arr = np.asarray(n_restrictions, dtype=np.int_)
	residual_df_arr = np.asarray(residual_df, dtype=np.int_)

	if f_values_arr.shape != n_restrictions_arr.shape:
		raise ResultsError(
			"f_values and n_restrictions must have equal shapes, got "
			f"{f_values_arr.shape} and {n_restrictions_arr.shape}"
		)

	if f_values_arr.shape != residual_df_arr.shape:
		raise ResultsError(
			"f_values and residual_df must have equal shapes, got "
			f"{f_values_arr.shape} and {residual_df_arr.shape}"
		)

	if np.any(n_restrictions_arr < 0):
		raise ResultsError("n_restrictions cannot contain negative values")

	if np.any(residual_df_arr <= 0):
		raise ResultsError("residual_df must be > 0 for every output, got " f"{residual_df_arr}")

	p_values = np.full(f_values_arr.shape, np.nan, dtype=np.float64)
	valid = (n_restrictions_arr > 0) & ~np.isnan(f_values_arr) & (f_values_arr >= 0)
	p_values[valid] = f.sf(np.maximum(f_values_arr[valid], 0.0), dfn=n_restrictions_arr[valid], dfd=residual_df_arr[valid])
	return p_values


def p_value_from_chi_square_test(
	test_values: NDArray[np.float64],
	df: NDArray[np.int_],
) -> NDArray[np.float64]:
	"""
	Convert per-output chi-square statistics to upper-tail p-values.
	"""
	values = np.asarray(test_values, dtype=np.float64)
	df_arr = np.asarray(df, dtype=np.int_)

	if values.shape != df_arr.shape:
		raise ResultsError(
			"test_values and df must have equal shapes, got "
			f"{values.shape} and {df_arr.shape}"
		)

	if np.any(df_arr < 0):
		raise ResultsError(f"Chi-square df cannot be negative, got {df_arr}")

	p_values = np.full(values.shape, np.nan, dtype=np.float64)
	valid = (df_arr > 0) & ~np.isnan(values) & (values >= 0)
	p_values[valid] = chi2.sf(np.maximum(values[valid], 0.0), df=df_arr[valid])
	return p_values


def p_value_from_wald_test(
	test_values: NDArray[np.float64],
	df: NDArray[np.int_],
) -> NDArray[np.float64]:
	"""Convert Wald statistics to chi-square p-values."""
	return p_value_from_chi_square_test(test_values, df)


def p_value_from_likelihood_ratio_test(
	test_values: NDArray[np.float64],
	df: NDArray[np.int_],
) -> NDArray[np.float64]:
	"""Convert likelihood-ratio statistics to chi-square p-values."""
	return p_value_from_chi_square_test(test_values, df)


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
