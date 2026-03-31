from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import f


def ensure_2d(array: NDArray[np.float64]) -> NDArray[np.float64]:
	"""Return a 2D float64 array, promoting 1D arrays to shape (n, 1)."""
	arr = np.asarray(array, dtype=np.float64)
	if arr.ndim == 1:
		arr = arr[:, np.newaxis]
	if arr.ndim != 2:
		raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
	return arr


def residual_sum_of_squares(
	y_true: NDArray[np.float64],
	y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
	"""Compute RSS per output: sum_t (y_true - y_pred)^2."""
	true_2d = ensure_2d(y_true)
	pred_2d = ensure_2d(y_pred)
	if true_2d.shape != pred_2d.shape:
		raise ValueError(
			f"y_true and y_pred must have the same shape, got {true_2d.shape} vs {pred_2d.shape}"
		)
	return np.sum((true_2d - pred_2d) ** 2, axis=0)


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
		raise ValueError("lag_order must be > 0")

	err_ref = np.asarray(error_ref, dtype=np.float64)
	err_base = np.asarray(error_base, dtype=np.float64)
	if err_ref.shape != err_base.shape:
		raise ValueError("error_ref and error_base must have the same shape")

	denominator = np.maximum(err_base * float(lag_order), np.finfo(np.float64).eps)
	numerator = (err_ref - err_base) * (float(n_samples) - float(rank))
	return numerator / denominator


def p_value_from_f_test(
	f_values: NDArray[np.float64],
	lag_order: int,
	df_denominator: float,
) -> NDArray[np.float64]:
	"""Convert F-statistics to p-values using upper tail: 1 - CDF(F)."""
	if lag_order <= 0:
		raise ValueError("lag_order must be > 0")
	df_den = max(float(df_denominator), 1.0)
	f_pos = np.maximum(np.asarray(f_values, dtype=np.float64), 0.0)
	return 1.0 - f.cdf(f_pos, lag_order, df_den)


def error_and_p_values(
	y_true: NDArray[np.float64],
	y_base_pred: NDArray[np.float64],
	y_ref_pred: NDArray[np.float64],
	lag_order: int,
	n_features: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
	"""
	Compute base/ref errors together with F-stat and p-values per output.

	Returns
	-------
	(base_error, ref_error, f_values, p_values)
	"""
	y_true_2d = ensure_2d(y_true)
	rss_base = residual_sum_of_squares(y_true_2d, y_base_pred)
	rss_ref = residual_sum_of_squares(y_true_2d, y_ref_pred)

	n_samples = y_true_2d.shape[0]
	rank = float(n_features) / float(lag_order)
	f_values = f_test_value(rss_ref, rss_base, lag_order=lag_order, rank=rank, n_samples=n_samples)
	p_values = p_value_from_f_test(f_values, lag_order=lag_order, df_denominator=n_samples - rank)

	# Legacy-compatible normalization used by previous implementation.
	n_outputs = y_true_2d.shape[1]
	scale = float(n_samples) * float(n_outputs)
	base_error = rss_base / scale
	ref_error = rss_ref / scale
	return base_error, ref_error, f_values, p_values
