from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
	from ..preprocessing.prepare_data import PreparedData

from .causality_matrix import CausalityMatrix
from .restoration import restore_removed_matrix
from .statistics import ensure_2d, residual_sum_of_squares

DofMethod = Literal["structural", "active_support"]


@dataclass
class ModelSnapshot:
	"""Stored outputs from one fitted model instance."""

	predictions: NDArray[np.float64]
	weights: NDArray[np.float64] | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparationContext:
	"""Context for preparing data for a single Granger causality analysis."""

	prepared_data: PreparedData
	metadata: dict[str, Any] | None = None

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
	"""Container of artifacts from a Granger causality analysis."""

	def __init__(
		self,
		effects: Optional[Iterable[str]] = None,
		causes: Optional[Iterable[str]] = None,
		mask: Optional[NDArray[np.int64]] = None,
		col_offsets: Optional[NDArray[np.int_]] = None,
		y_true: Optional[NDArray[np.float64]] = None,
		X_true: Optional[NDArray[np.float64]] = None,
	) -> None:
		self.effects = list(effects or [])
		self.causes = list(causes or [])
		self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.empty((0, 0), dtype=bool)
		self.col_offsets = np.asarray(col_offsets, dtype=np.int_) if col_offsets is not None else np.asarray([], dtype=np.int_)
		self.y_true = ensure_2d(np.asarray(y_true, dtype=np.float64)) if y_true is not None else np.empty((0, 0), dtype=np.float64)
		self.X_true = ensure_2d(np.asarray(X_true, dtype=np.float64)) if X_true is not None else np.empty((0, 0), dtype=np.float64)
		self.dof_method: DofMethod = "structural"
		self.base_snapshot: ModelSnapshot | None = None
		self.reference_snapshots: dict[str, ModelSnapshot] = {}
		self._metric_cache: dict[Any, CausalityMatrix] = {}
		self._rss_cache: dict[str, NDArray[np.float64]] = {}
		self.preparation_context: Optional[PreparationContext] = None

	@property
	def base_predictions(self) -> NDArray[np.float64] | None:
		return None if self.base_snapshot is None else self.base_snapshot.predictions

	@property
	def base_weights(self) -> NDArray[np.float64] | None:
		return None if self.base_snapshot is None else self.base_snapshot.weights

	@property
	def ref_predictions(self) -> dict[str, NDArray[np.float64]]:
		return {name: snap.predictions for name, snap in self.reference_snapshots.items()}

	@property
	def ref_weights(self) -> dict[str, NDArray[np.float64]]:
		return {name: snap.weights for name, snap in self.reference_snapshots.items() if snap.weights is not None}

	@property
	def n_samples(self) -> int:
		return int(self.y_true.shape[0]) if self.y_true.size else 0

	@property
	def p_value(self) -> pd.DataFrame:
		return self.evaluate("p_value", restore_removed=True).data

	@property
	def f_test(self) -> pd.DataFrame:
		return self.evaluate("f_statistic", restore_removed=True).data

	@property
	def wald_test(self) -> pd.DataFrame:
		return self.evaluate("wald_test", restore_removed=True).data

	@property
	def likelihood_ratio(self) -> pd.DataFrame:
		return self.evaluate("likelihood_ratio", restore_removed=True).data

	@property
	def sign(self) -> pd.DataFrame:
		return self._sign_matrix()

	def set_dof_method(self, method: DofMethod) -> None:
		"""Select whether test degrees of freedom use structure or active weights."""
		if method not in {"structural", "active_support"}:
			raise ValueError("dof method must be 'structural' or 'active_support'")
		self.dof_method = method

	def _weights_as_feature_output(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Normalize weights to the ``effects x features`` orientation."""
		weights_arr = np.asarray(weights, dtype=np.float64)
		if weights_arr.ndim != 2:
			raise ValueError(f"Expected 2D weights, got shape {weights_arr.shape}")
		if weights_arr.shape[0] == len(self.effects) and weights_arr.shape[1] != len(self.effects):
			return weights_arr
		if weights_arr.shape[1] == len(self.effects):
			return weights_arr.T
		return weights_arr

	def _effect_cause_matrix(self, values: NDArray[np.float64], *, name: str) -> CausalityMatrix:
		"""Wrap numeric metric values in an effects-by-causes matrix."""
		values_arr = np.asarray(values, dtype=np.float64)
		if values_arr.ndim == 1:
			values_arr = values_arr[:, np.newaxis]
		if values_arr.shape == (len(self.effects), len(self.causes)):
			matrix = pd.DataFrame(values_arr, index=self.effects, columns=self.causes)
		else:
			filled = np.full((len(self.effects), len(self.causes)), np.nan, dtype=np.float64)
			filled[: values_arr.shape[0], : values_arr.shape[1]] = values_arr[: len(self.effects), : len(self.causes)]
			matrix = pd.DataFrame(filled, index=self.effects, columns=self.causes)
		return CausalityMatrix(data=matrix, name=name)

	def set_preparation_context(self, context: PreparationContext) -> None:
		"""Attach preprocessing metadata used to restore removed variables."""
		self.preparation_context = context

	def set_base_snapshot(
			self,
			base_predictions: NDArray[np.float64],
			base_weights: NDArray[np.float64],
			metadata: Optional[dict[str, Any]] = None,
		) -> None:
			"""Store predictions and weights from the unrestricted base model."""
			self.base_snapshot = ModelSnapshot(predictions=ensure_2d(base_predictions), weights=np.asarray(base_weights, dtype=np.float64), metadata=metadata or {})
			self._rss_cache.pop("base", None)
			self._metric_cache.clear()

	def set_reference_snapshot(
			self,
			cause: str,
			reference_predictions: NDArray[np.float64],
			reference_weights: NDArray[np.float64],
			metadata: Optional[dict[str, Any]] = None,
		) -> None:
			"""Store predictions and weights from a model with one cause omitted."""
			self.reference_snapshots[cause] = ModelSnapshot(
				predictions=ensure_2d(reference_predictions),
				weights=np.asarray(reference_weights, dtype=np.float64),
				metadata=metadata or {},
			)
			self._rss_cache.pop(f"reference:{cause}", None)
			self._metric_cache.clear()

	def n_restrictions_for(self, cause: str) -> NDArray[np.int_]:
		"""Count tested coefficients for one cause separately for every effect."""
		if cause not in self.causes:
			raise ValueError(f"Unknown cause '{cause}'")
		if self.mask.size == 0:
			return np.zeros(len(self.effects), dtype=np.int_)
		cause_index = self.causes.index(cause)
		if cause_index >= len(self.col_offsets) - 1:
			return np.zeros(len(self.effects), dtype=np.int_)
		start = int(self.col_offsets[cause_index])
		end = int(self.col_offsets[cause_index + 1])
		mask_block = self.mask[:, start:end]
		if self.dof_method == "active_support" and self.base_snapshot is not None and self.base_snapshot.weights is not None:
			weight_block = np.abs(self._weights_as_feature_output(self.base_snapshot.weights)[:, start:end])
			mask_block = mask_block & (weight_block > 0.0)
		return np.asarray(mask_block.sum(axis=1), dtype=np.int_)

	def n_unrestricted_parameters(self) -> int:
		"""Return the parameter count used for residual degrees of freedom."""
		if self.base_snapshot is not None and self.base_snapshot.weights is not None:
			weights = self._weights_as_feature_output(self.base_snapshot.weights)
			return int(max(weights.shape[1], 1))
		if self.X_true.size:
			return int(max(self.X_true.shape[1], 1))
		return 1

	def base_rss_per_output(self) -> NDArray[np.float64]:
		"""Compute unrestricted-model RSS separately for every model output."""
		if self.base_snapshot is None:
			raise ValueError("Base snapshot is not available")
		if "base" not in self._rss_cache:
			self._rss_cache["base"] = residual_sum_of_squares(
				self.y_true,
				self.base_snapshot.predictions,
			)
		return self._rss_cache["base"]

	def reference_rss_per_output(self, cause: str) -> NDArray[np.float64]:
		"""Compute omitted-cause RSS separately for every model output."""
		if cause not in self.reference_snapshots:
			raise ValueError(f"Reference snapshot for cause '{cause}' is not available")
		cache_key = f"reference:{cause}"
		if cache_key not in self._rss_cache:
			snapshot = self.reference_snapshots[cause]
			self._rss_cache[cache_key] = residual_sum_of_squares(
				self.y_true,
				snapshot.predictions,
			)
		return self._rss_cache[cache_key]

	def make_matrix(
		self,
		values: NDArray[np.float64],
		name: str = "result",
		metadata: Optional[dict[str, Any]] = None,
	) -> CausalityMatrix:
		"""Build an effects-by-causes matrix and attach metric metadata."""
		matrix = self._effect_cause_matrix(values, name=name)
		if metadata:
			matrix.metadata.update(metadata)
		return matrix

	def evaluate(
			self,
			metric: str | Callable[..., CausalityMatrix] = "f_statistic",
			*,
			restore_removed: bool = True,
			cache: bool = True,
			force: bool = False,
			**kwargs: Any,
		) -> CausalityMatrix:
			"""Evaluate and optionally cache a metric, restoring removed variables."""
			from .metrics import resolve_metric

			metric_func = resolve_metric(metric)
			cache_key = (metric, tuple(sorted(kwargs.items()))) if kwargs else metric
			if cache and cache_key in self._metric_cache and not force:
				result = self._metric_cache[cache_key]
			else:
				result = metric_func(self, **kwargs)
				if cache:
					self._metric_cache[cache_key] = result
			if restore_removed and self.preparation_context is not None:
				result = CausalityMatrix(
					data=restore_removed_matrix(
						result,
						effects=self.effects,
						causes=self.causes,
						removed_variables=self.preparation_context.removed_variables,
					),
					name=result.name,
					metadata=result.metadata,
				)
			return result

	def result(
			self,
			threshold: float = 0.01,
			with_sign: bool = False,
			metric: str | Callable[..., CausalityMatrix] = "f_test_p_value",
		) -> pd.DataFrame:
			"""Return a binary thresholded causality table, optionally signed."""
			metric_matrix = self.evaluate(metric, restore_removed=True)
			result_df = metric_matrix.data.copy()
			binary = (result_df < threshold).astype(int)
			if with_sign:
				binary = binary.mul(self.sign)
			return binary

	def _sign_matrix(self) -> pd.DataFrame:
		"""Extract one causal sign per effect-cause pair from base weights."""
		if self.base_snapshot is None or self.base_snapshot.weights is None:
			return pd.DataFrame(np.zeros((len(self.effects), len(self.causes))), index=self.effects, columns=self.causes, dtype=np.float64)
		weights_matrix = self._weights_as_feature_output(self.base_snapshot.weights)
		sign_values = np.zeros((len(self.effects), len(self.causes)), dtype=np.float64)
		for cause_index, cause in enumerate(self.causes):
			if cause_index >= len(self.col_offsets) - 1:
				continue
			start = int(self.col_offsets[cause_index])
			end = int(self.col_offsets[cause_index + 1])
			block = weights_matrix[:, start:end]
			sum_coefficients = block.sum(axis=1)
			sign_values[:, cause_index] = np.sign(sum_coefficients)
			sign_values[np.abs(sum_coefficients) < 1e-12, cause_index] = 1.0
		return pd.DataFrame(sign_values, index=self.effects, columns=self.causes)

	def __repr__(self) -> str:
		return (
			f"GrangerAnalysisResults(effects={self.effects}, causes={self.causes}, "
			f"dof_method={self.dof_method!r})"
		)
		