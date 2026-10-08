from dataclasses import dataclass, field
from collections.abc import Callable
from modulefinder import test
from modulefinder import test
from typing import Iterable, Optional, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..preprocessing.prepare_data import PreparedData
from .causality_matrix import CausalityMatrix
from .metrics import resolve_metric
from .statistics import (
	ensure_2d,
	error_values,
	estimate_masked_weight_covariance,
	f_test_value,
	likelihood_ratio_test_value,
	p_value_from_chi_square_test,
	p_value_from_f_test,
	wald_test_value,
)


@dataclass
class ModelSnapshot:
	"""Stored outputs from one fitted model instance."""

	predictions: NDArray[np.float64]
	weights: NDArray[np.float64]| None = None

	metadata: dict[str,Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparationContext:
	""" Context for preparing data for a single Granger causality analysis. 
	Contains the full PreparedData and objects needed for interpretation, reproduction, and extended analysis of the results. 
	"""
	prepared_data: PreparedData
	metadata: dict [str,Any] | None = None

	@property
	def stationarity_transformer(self):
		return self.prepared_data.stationarity_transformer

	@property
	def lag_engine(self):
		return self.prepared_data.lag_engine
	@property
	def x_scaler(self):
		return self.prepared_data.x_scaler
	@property
	def y_scaler(self):
		return self.prepared_data.y_scaler

	@property
	def removed_variables(self) -> dict[str, dict[str, Any]]:
		return self.prepared_data.removed_variable_info


class GrangerAnalysisResults:
	"""
	Container of artifacts from a Granger causality analysis.

	Metrics are calculated after using evaluate()
	"""
	def __init__(
		self, 
		effects: Iterable[str],
		causes: Iterable[str],
		mask: NDArray[np.int64], col_offsets: NDArray[np.int_],
		y_true: NDArray[np.float64],
		X_true: NDArray[np.float64],
	) -> None :
		self.effects = list(effects)
		self.causes = list(causes)
		self.mask = np.asarray(mask, dtype=bool)
		self.col_offsets = np.asarray(col_offsets, dtype=np.int_)
		self.y_true = ensure_2d(np.asarray(y_true, dtype=np.float64))
		self.X_true = ensure_2d(np.asarray(X_true, dtype=np.float64))

		self.base_snapshot: ModelSnapshot | None = None
		self.reference_snapshots: dict[str, ModelSnapshot] = {}

		# Opcjonalnie: mały cache wyników obliczonych metryk.
		self._metric_cache: dict[tuple, CausalityMatrix] = {}
		self.preparation_context: Optional[PreparationContext] = None

	def set_preparation_context(self, context: PreparationContext) -> None:
		"""Store the preparation context for later reference."""
		self.preparation_context = context

	def set_base_snapshot(
			self,
			base_predictions: NDArray[np.float64],
			base_weights: NDArray[np.float64],
			metadata: Optional[dict[str, Any]] = None
		) -> None:
			"""Store base model predictions and weights for later use in causality tests."""
			self.base_snapshot = ModelSnapshot(predictions=base_predictions, weights=base_weights, metadata=metadata or {})

	def set_reference_snapshot(
			self,
			cause: str,
			reference_predictions: NDArray[np.float64],
			reference_weights: NDArray[np.float64],
			metadata: Optional[dict[str, Any]] = None
		) -> None:
			"""Store reference model predictions and weights for a specific cause."""
			self.reference_snapshots[cause] = ModelSnapshot(
				predictions=reference_predictions, 
				weights=reference_weights, 
				metadata=metadata or {}
			)

	def evaluate(
			self,
			metric: str | Callable[..., CausalityMatrix]= "f_statistic",
			*,
			restore_removed: bool = True,
			cache: bool = True,
			force: bool = False,
			**kwargs,
		) -> CausalityMatrix:
			"""Evaluate a metric and return the resulting causality matrix.
	
			Args:
				metric: The name of the metric to evaluate or a callable that computes it.
				restore_removed: If True, restore rows/columns for removed variables.
				cache: If True, cache the result for future calls with the same parameters.
				force: If True, force re-evaluation of the metric even if it's already cached.
				**kwargs: Additional keyword arguments to pass to the metric function.

			Returns:
				The resulting causality matrix.
			"""
			# Implementation for evaluating the metric would go here.
			metric_func = resolve_metric(metric)
			if cache:
				cache_key = metric
				if cache_key in self._metric_cache.keys() and not force:
					return self._metric_cache[cache_key]
				result = metric_func(self, **kwargs)
				self._metric_cache[cache_key] = result
				return result
			pass

	def result(
			self, 
			threshold: float = 0.01, 
			with_sign: bool = False,
			metric: str | Callable[..., CausalityMatrix] = "f_test_p_value"
		) -> pd.DataFrame:
		"""Return binary/signed causality matrix derived from p-value table."""
		
		if metric not in self._metric_cache.keys():
			metric_func = resolve_metric(metric)
			metric_matrix = metric_func(self)
			self._metric_cache[metric] = metric_matrix
		else:
			metric_matrix = self._metric_cache[metric]

		if with_sign:
			if "sign" not in self._metric_cache.keys():
				sign_matrix = resolve_metric("sign")(self)
				self._metric_cache["sign"] = sign_matrix
			else:
				sign_matrix = self._metric_cache["sign"]
			return metric_matrix.threshold(threshold=threshold) * sign_matrix
		else:		
			return metric_matrix.threshold(threshold=threshold)
		