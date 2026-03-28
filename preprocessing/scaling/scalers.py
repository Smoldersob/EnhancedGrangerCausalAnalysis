"""
Scaling implementations compatible with :class:`complex_granger_analysis.core.protocols.Scaler`.

The module provides common deterministic scaling strategies for multivariate
time series data represented as 2D arrays: ``(n_samples, n_features)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _as_2d_float64(data: NDArray[np.float64]) -> NDArray[np.float64]:
	"""Return a float64 2D array copy suitable for numerical scaling."""
	arr = np.asarray(data, dtype=np.float64)
	if arr.ndim != 2:
		raise ValueError(
			f"Expected 2D array of shape (n_samples, n_features), got ndim={arr.ndim}"
		)
	return arr.copy()


class _BaseScaler:
	"""Shared fit-state handling for concrete scalers."""

	_fitted: bool = False

	def _ensure_fitted(self) -> None:
		if not self._fitted:
			raise RuntimeError("Scaler is not fitted. Call fit_transform first.")


class StandardScaler(_BaseScaler):
	"""Column-wise z-score scaling: ``(x - mean) / std``."""

	def __init__(self, eps: float = 1e-12):
		self.eps = float(eps)
		self.mean_: Optional[NDArray[np.float64]] = None
		self.scale_: Optional[NDArray[np.float64]] = None

	def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		x = _as_2d_float64(data)
		self.mean_ = x.mean(axis=0)
		std = x.std(axis=0)
		self.scale_ = np.where(std > self.eps, std, 1.0)
		self._fitted = True
		return (x - self.mean_) / self.scale_

	def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		return (x - self.mean_) / self.scale_

	def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		return x * self.scale_ + self.mean_


class MinMaxScaler(_BaseScaler):
	"""Column-wise min-max scaling to a given range, default ``[0, 1]``."""

	def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-12):
		lo, hi = feature_range
		if hi <= lo:
			raise ValueError("feature_range must satisfy max > min")
		self.feature_range = (float(lo), float(hi))
		self.eps = float(eps)
		self.data_min_: Optional[NDArray[np.float64]] = None
		self.data_max_: Optional[NDArray[np.float64]] = None
		self.data_range_: Optional[NDArray[np.float64]] = None

	def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		x = _as_2d_float64(data)
		self.data_min_ = x.min(axis=0)
		self.data_max_ = x.max(axis=0)
		rng = self.data_max_ - self.data_min_
		self.data_range_ = np.where(rng > self.eps, rng, 1.0)
		self._fitted = True

		lo, hi = self.feature_range
		x_std = (x - self.data_min_) / self.data_range_
		return x_std * (hi - lo) + lo

	def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		lo, hi = self.feature_range
		x_std = (x - self.data_min_) / self.data_range_
		return x_std * (hi - lo) + lo

	def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		lo, hi = self.feature_range
		x_std = (x - lo) / (hi - lo)
		return x_std * self.data_range_ + self.data_min_


class RobustScaler(_BaseScaler):
	"""Column-wise robust scaling using median and IQR."""

	def __init__(self, eps: float = 1e-12):
		self.eps = float(eps)
		self.center_: Optional[NDArray[np.float64]] = None
		self.scale_: Optional[NDArray[np.float64]] = None

	def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		x = _as_2d_float64(data)
		q25 = np.percentile(x, 25.0, axis=0)
		q75 = np.percentile(x, 75.0, axis=0)
		iqr = q75 - q25

		self.center_ = np.median(x, axis=0)
		self.scale_ = np.where(iqr > self.eps, iqr, 1.0)
		self._fitted = True
		return (x - self.center_) / self.scale_

	def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		return (x - self.center_) / self.scale_

	def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		return x * self.scale_ + self.center_


class MaxAbsScaler(_BaseScaler):
	"""Column-wise scaling by maximum absolute value."""

	def __init__(self, eps: float = 1e-12):
		self.eps = float(eps)
		self.max_abs_: Optional[NDArray[np.float64]] = None

	def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		x = _as_2d_float64(data)
		max_abs = np.max(np.abs(x), axis=0)
		self.max_abs_ = np.where(max_abs > self.eps, max_abs, 1.0)
		self._fitted = True
		return x / self.max_abs_

	def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		return x / self.max_abs_

	def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		x = _as_2d_float64(data)
		return x * self.max_abs_


class IdentityScaler(_BaseScaler):
	"""No-op scaler useful for pipelines that require a scaler interface."""

	def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		x = _as_2d_float64(data)
		self._fitted = True
		return x

	def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		return _as_2d_float64(data)

	def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
		self._ensure_fitted()
		return _as_2d_float64(data)



