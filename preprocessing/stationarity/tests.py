from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def _as_clean_float_array(series: pd.Series) -> np.ndarray:
	values = pd.Series(series).astype(float).dropna().to_numpy()
	return values


def _is_constant(values: np.ndarray) -> bool:
	return values.size == 0 or np.nanvar(values) == 0.0


def _difference_once(values: np.ndarray) -> np.ndarray:
	if values.size <= 1:
		return np.array([], dtype=float)
	return np.diff(values)


def static_adfuller_order(series: pd.Series, maxlag: int = 5, alpha: float = 0.05) -> int:
	"""Return the differencing order that best satisfies ADF stationarity.

	Strategy:
	- Check orders from 0 to ``maxlag``.
	- Return first order with p-value <= ``alpha``.
	- If none is stationary, return order with the smallest p-value.
	"""
	values = _as_clean_float_array(series)
	if _is_constant(values):
		return 0

	best_order = 0
	best_p = np.inf

	for order in range(maxlag + 1):
		if values.size < 4 or _is_constant(values):
			break

		try:
			p_value = float(adfuller(values, autolag="AIC")[1])
		except Exception:
			p_value = 1.0

		if p_value <= alpha:
			return order

		if p_value < best_p:
			best_p = p_value
			best_order = order

		values = _difference_once(values)

	return best_order


def static_kpss_order(series: pd.Series, maxlag: int = 5, alpha: float = 0.05) -> int:
	"""Return the differencing order that best satisfies KPSS stationarity.

	Strategy:
	- Check orders from 0 to ``maxlag``.
	- Return first order with p-value >= ``alpha``.
	- If none is stationary, return order with the largest p-value.
	"""
	values = _as_clean_float_array(series)
	if _is_constant(values):
		return 0

	best_order = 0
	best_p = -np.inf

	for order in range(maxlag + 1):
		if values.size < 4 or _is_constant(values):
			break

		try:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=InterpolationWarning)
				p_value = float(kpss(values, regression="ct", nlags="auto")[1])
		except Exception:
			p_value = 0.0

		if p_value >= alpha:
			return order

		if p_value > best_p:
			best_p = p_value
			best_order = order

		values = _difference_once(values)

	return best_order


def apply_differencing(series: pd.Series, order: int) -> pd.Series:
	"""Apply differencing of given order and preserve index alignment."""
	if order <= 0:
		return pd.Series(series).copy()

	out = pd.Series(series).astype(float).copy()
	for _ in range(order):
		out = out.diff()
	return out

