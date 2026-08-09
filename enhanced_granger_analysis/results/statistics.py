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
    covariance = np.zeros(
        (n_outputs, n_features, n_features),
        dtype=np.float64,
    )

    for out_idx in range(n_outputs):
        active_idx = np.flatnonzero(mask_arr[out_idx])

        if active_idx.size == 0:
            raise ResultsError(
                f"Output {out_idx} has no active slope coefficients"
            )

        x_active = x_arr[:, active_idx]

        if fit_intercept:
            design = np.column_stack(
                [np.ones(n_samples, dtype=np.float64), x_active]
            )
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

        sigma2 = (
            np.sum(residuals[:, out_idx] ** 2)
            / float(residual_df)
        )

        covariance_active = sigma2 * np.linalg.inv(design.T @ design)

        if fit_intercept:
            covariance_active = covariance_active[1:, 1:]

        covariance[out_idx][
            np.ix_(active_idx, active_idx)
        ] = covariance_active

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
	n_outputs = y_true_2d.shape[1]
	scale = float(n_samples) * float(n_outputs)
	base_error = rss_base / scale
	ref_error = rss_ref / scale
	return base_error, ref_error

def wald_test_value(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
    coefficient_mask: NDArray[np.float64],
    start: int,
    end: int,
) -> NDArray[np.float64]:
    """Return one Wald statistic per output; NaN if structurally untestable."""
    weights_arr = np.asarray(weights, dtype=np.float64)
    cov_arr = np.asarray(covariance, dtype=np.float64)
    mask_arr = np.asarray(coefficient_mask, dtype=bool)

    n_outputs, n_features = weights_arr.shape

    if mask_arr.shape != weights_arr.shape:
        raise ResultsError(
            f"Mask shape {mask_arr.shape} does not match weights shape "
            f"{weights_arr.shape}"
        )

    if cov_arr.shape != (n_outputs, n_features, n_features):
        raise ResultsError(
            "Invalid covariance shape: "
            f"expected {(n_outputs, n_features, n_features)}, "
            f"got {cov_arr.shape}"
        )

    if not (0 <= start < end <= n_features):
        raise ResultsError(
            f"Invalid tested coefficient block [{start}:{end}]"
        )

    cause_indices = np.arange(start, end)
    statistics = np.full(n_outputs, np.nan, dtype=np.float64)

    for out_idx in range(n_outputs):
        tested_idx = cause_indices[
            mask_arr[out_idx, cause_indices]
        ]

        if tested_idx.size == 0:
            continue

        beta_block = weights_arr[out_idx, tested_idx]

        cov_block = cov_arr[out_idx][
            np.ix_(tested_idx, tested_idx)
        ]
        cov_block = 0.5 * (cov_block + cov_block.T)

        if np.linalg.matrix_rank(cov_block) != tested_idx.size:
            raise ResultsError(
                f"Singular Wald covariance for output {out_idx}, "
                f"tested indices={tested_idx.tolist()}"
            )

        statistics[out_idx] = float(
            beta_block @ np.linalg.solve(cov_block, beta_block)
        )

    return statistics

def f_test_value(
    error_ref: NDArray[np.float64],
    error_base: NDArray[np.float64],
    n_restrictions: NDArray[np.int64],
    n_unrestricted_parameters: NDArray[np.int64],
    n_samples: int,
) -> NDArray[np.float64]:
    """
    F test for nested restricted and unrestricted OLS models.

    F = ((RSS_restricted - RSS_unrestricted) / q) /
        (RSS_unrestricted / (n_samples - k_unrestricted))

    Parameters
    ----------
    error_ref:
        Restricted-model error/RSS per output.
    error_base:
        Unrestricted-model error/RSS per output.
    n_restrictions:
        Number of tested zero-coefficient restrictions, q.
    n_unrestricted_parameters:
        Total number of fitted parameters in one unrestricted equation,
        including intercept if present, k_U.
    n_samples:
        Number of effective observations after lag construction.
    """
    if np.any(n_restrictions < 0):
        raise ResultsError(
            f"n_restrictions must be >= 0, got {n_restrictions}"
        )

    residual_df = n_samples - n_unrestricted_parameters
    
    if np.any(residual_df <= 0):
        raise ResultsError(
            "Unrestricted residual degrees of freedom must be > 0, got "
            f"n_samples={n_samples}, "
            f"n_unrestricted_parameters={n_unrestricted_parameters}"
        )

    err_ref = np.asarray(error_ref, dtype=np.float64)
    err_base = np.asarray(error_base, dtype=np.float64)

    if err_ref.shape != err_base.shape:
        raise ResultsError(
            "error_ref and error_base must have the same shape"
        )
    
    numerator = np.max([err_ref - err_base,err_base*0],axis=0) / np.maximum(n_restrictions, np.finfo(np.float64).eps)
    denominator = err_base / residual_df
    return numerator / np.maximum(denominator, np.finfo(np.float64).eps)

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
	return float(n_samples) * np.log(ratio)


def p_value_from_f_test(
    f_values: NDArray[np.float64],
    n_restrictions: NDArray[np.int_],
    residual_df: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Convert per-output F statistics to upper-tail p-values.

    Outputs with n_restrictions == 0 are structurally untestable and
    receive np.nan.
    """
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
        raise ResultsError(
            "n_restrictions cannot contain negative values"
        )

    if np.any(residual_df_arr <= 0):
        raise ResultsError(
            "residual_df must be > 0 for every output, got "
            f"{residual_df_arr}"
        )

    p_values = np.full(
        f_values_arr.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        (n_restrictions_arr > 0)
        & np.isfinite(f_values_arr)
    )

    p_values[valid] = f.sf(
        np.maximum(f_values_arr[valid], 0.0),
        dfn=n_restrictions_arr[valid],
        dfd=residual_df_arr[valid],
    )

    return p_values

def p_value_from_chi_square_test(
    test_values: NDArray[np.float64],
    df: NDArray[np.int_],
) -> NDArray[np.float64]:
    """
    Convert per-output chi-square statistics to upper-tail p-values.

    df == 0 means that the corresponding relation is structurally
    excluded by the fixed mask, so the result is np.nan.
    """
    values = np.asarray(test_values, dtype=np.float64)
    df_arr = np.asarray(df, dtype=np.int_)

    if values.shape != df_arr.shape:
        raise ResultsError(
            "test_values and df must have equal shapes, got "
            f"{values.shape} and {df_arr.shape}"
        )

    if np.any(df_arr < 0):
        raise ResultsError(
            f"Chi-square df cannot be negative, got {df_arr}"
        )

    p_values = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        (df_arr > 0)
        & np.isfinite(values)
    )

    p_values[valid] = chi2.sf(
        np.maximum(values[valid], 0.0),
        df=df_arr[valid],
    )

    return p_values