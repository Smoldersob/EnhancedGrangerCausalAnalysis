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

def estimate_weight_covariance(
	x_train: NDArray[np.float64],
	y_true: NDArray[np.float64],
	y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
	"""Estimate per-output coefficient covariance from a design matrix."""
	x_arr = np.asarray(x_train, dtype=np.float64)
	if x_arr.ndim != 2:
		raise ResultsError(f"Expected 2D design matrix, got shape {x_arr.shape}")

	y_true_2d = ensure_2d(y_true)
	y_pred_2d = ensure_2d(y_pred)
	if y_true_2d.shape != y_pred_2d.shape:
		raise ResultsError(
			f"y_true and y_pred must have the same shape, got {y_true_2d.shape} vs {y_pred_2d.shape}"
		)
	if x_arr.shape[0] != y_true_2d.shape[0]:
		raise ResultsError(
			f"Design matrix and targets must have the same number of rows, got {x_arr.shape[0]} vs {y_true_2d.shape[0]}"
		)

	xtx_inv = np.linalg.pinv(x_arr.T @ x_arr)
	residuals = y_true_2d - y_pred_2d
	n_samples = x_arr.shape[0]
	n_features = x_arr.shape[1]
	dof = max(n_samples - n_features, 1)
	sigma2 = np.sum(residuals**2, axis=0) / float(dof)
	return sigma2[:, np.newaxis, np.newaxis] * xtx_inv[np.newaxis, :, :]

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
	start: int,
	end: int,
) -> NDArray[np.float64]:
	"""Compute Wald statistic vector from coefficient blocks and covariance matrices."""
	weights_arr = np.asarray(weights, dtype=np.float64)
	if weights_arr.ndim != 2:
		raise ResultsError(f"Expected 2D weights matrix, got shape {weights_arr.shape}")
	if start < 0 or end < 0 or end < start:
		raise ResultsError("Invalid coefficient block boundaries")
	if end > weights_arr.shape[1]:
		raise ResultsError(
			f"Coefficient block end {end} exceeds available features {weights_arr.shape[1]}"
		)

	cov_arr = np.asarray(covariance, dtype=np.float64)
	if cov_arr.ndim == 2:
		cov_by_output = np.broadcast_to(cov_arr, (weights_arr.shape[0],) + cov_arr.shape)
	elif cov_arr.ndim == 3 and cov_arr.shape[0] == weights_arr.shape[0]:
		cov_by_output = cov_arr
	else:
		raise ResultsError(
			f"Covariance must be 2D or 3D with the first dimension matching outputs, got shape {cov_arr.shape}"
		)

	block_size = end - start
	if block_size == 0:
		return np.zeros(weights_arr.shape[0], dtype=np.float64)

	statistics = np.zeros(weights_arr.shape[0], dtype=np.float64)
	for out_idx in range(weights_arr.shape[0]):
		beta = weights_arr[out_idx, start:end]
		block_cov = cov_by_output[out_idx][start:end, start:end]
		if block_cov.size == 0:
			continue
		inv_cov = np.linalg.pinv(block_cov)
		statistics[out_idx] = float(beta @ inv_cov @ beta.T)

	return statistics

def f_test_value(
	error_ref: NDArray[np.float64],
	error_base: NDArray[np.float64],
	lag_order: int,
	rank: float,
	n_samples: int,
) -> NDArray[np.float64]:
	"""
	Compute F-statistic vector from reference and base model errors.

	This follows the scheme used in the legacy implementation:
	F = (error_ref - error_base) * (n - rank) / (error_base * lag_order)
	"""
	if lag_order <= 0:
		raise ResultsError("lag_order must be > 0")

	err_ref = np.asarray(error_ref, dtype=np.float64)
	err_base = np.asarray(error_base, dtype=np.float64)
	if err_ref.shape != err_base.shape:
		raise ResultsError("error_ref and error_base must have the same shape")

	denominator = np.maximum(err_base * float(lag_order), np.finfo(np.float64).eps)
	numerator = (err_ref - err_base) * (float(n_samples) - float(rank))
	return numerator / denominator


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
	lag_order: int,
	df_denominator: float,
) -> NDArray[np.float64]:
	"""Convert F-statistics to p-values using upper tail: 1 - CDF(F)."""
	if lag_order <= 0:
		raise ResultsError("lag_order must be > 0")
	df_den = max(float(df_denominator), 1.0)
	f_pos = np.maximum(np.asarray(f_values, dtype=np.float64), 0.0)
	return 1.0 - f.cdf(f_pos, lag_order, df_den)


def p_value_from_chi_square_test(
	test_values: NDArray[np.float64],
	df: float,
) -> NDArray[np.float64]:
	"""Convert chi-square-style statistics to upper-tail p-values."""
	df_safe = max(float(df), 1.0)
	stat_pos = np.maximum(np.asarray(test_values, dtype=np.float64), 0.0)
	return 1.0 - chi2.cdf(stat_pos, df_safe)


