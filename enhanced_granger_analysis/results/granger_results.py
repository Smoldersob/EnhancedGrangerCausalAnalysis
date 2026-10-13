"""Container objects and cached result evaluation for Granger causality analysis.

The module stores model snapshots, preprocessing metadata, and derived
causality matrices computed from restricted and unrestricted model fits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
	from ..preprocessing.prepare_data import PreparedData, RemovedVariable, VariableRoles

from .causality_matrix import CausalityMatrix
from .restoration import restore_removed_matrix
from .statistics import ensure_2d, residual_sum_of_squares

DofMethod = Literal["structural", "active_support"]


@dataclass
class ModelSnapshot:
	"""
	Snapshot of one fitted model run.
	Stores predictions, optional coefficient weights, and arbitrary
	metadata associated with a single unrestricted or restricted model fit.
	"""

	predictions: NDArray[np.float64]
	weights: NDArray[np.float64] | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparationContext:
	"""
	Encapsulates preprocessing state used during a Granger test.
	The context exposes the original prepared-data objects and the
	metadata needed to restore variables removed during feature preparation.
	"""

	prepared_data: PreparedData
	metadata: dict[str, Any] | None = None

	@property
	def stationarity_transformer(self):
		"""Return the transformer used to stabilize the stationarity step.

		Returns: Transformer instance applied to the prepared data pipeline before
		model fitting.
		"""
		return self.prepared_data.stationarity_transformer

	@property
	def lag_engine(self):
		"""Return the lag-generation engine used by the analysis.

		Returns: Lag-engine object producing time-lagged regression features.
		"""
		return self.prepared_data.lag_engine

	@property
	def x_scaler(self):
		"""Return the feature scaler attached to the model inputs.

		Returns: Scaling transformer applied to explanatory variables.
		"""
		return self.prepared_data.x_scaler

	@property
	def y_scaler(self):
		"""Return the target scaler attached to the response variables.

		Returns: Scaling transformer applied to dependent variables.
		"""
		return self.prepared_data.y_scaler

	@property
	def removed_variables(self) -> dict[str, RemovedVariable]:
		"""Return typed metadata for variables removed during preparation."""
		return self.prepared_data.removed_variables

	@property
	def requested_roles(self) -> VariableRoles:
		"""
		Return the variable roles requested by the user.
		"""
		return self.prepared_data.requested_roles

	@property
	def effective_roles(self) -> VariableRoles:
		"""
		Return the variable roles actually used by the analysis after preprocessing.
		"""
		return self.prepared_data.effective_roles

class GrangerAnalysisResults:
	"""
	Container for model outputs and Granger-causality test results.
	This object stores the fitted unrestricted model, the restricted
	reference models for each removed cause, and the cached matrices that
	represent statistical metrics such as p-values, F-statistics, and likelihood
	ratios.
	"""

	def __init__(
		self,
		effects: Optional[Iterable[str]] = None,
		causes: Optional[Iterable[str]] = None,
		mask: Optional[NDArray[np.int64]] = None,
		cause_slices: Optional[Mapping[str, slice]] = None,
		y_true: Optional[NDArray[np.float64]] = None,
		X_true: Optional[NDArray[np.float64]] = None,
	) -> None:
		"""Initialize an empty result store for a Granger causality analysis.

		Args:
			effects: Names of the response variables being evaluated.
			causes: Names of the candidate causal drivers being tested.
			mask: Boolean mask describing which coefficient slots are active.
			cause_slices: Mapping from cause name to the corresponding slice in the feature matrix.
			y_true: Observed response matrix used to compute residual sums of
				squares.
			X_true: Original explanatory data matrix used as a fallback source
				for unrestricted parameter counts.
		"""
		self.effects = list(effects or [])
		self.causes = list(causes or [])
		self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.empty((0, 0), dtype=bool)
		self.cause_slices = dict(cause_slices or {})
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
		"""Return predictions from the unrestricted base model.

		Returns: Base-model predictions as a 2D array or None when unavailable.
		"""
		return None if self.base_snapshot is None else self.base_snapshot.predictions

	@property
	def base_weights(self) -> NDArray[np.float64] | None:
		"""Return coefficient weights from the unrestricted base model.

		Returns: Weight matrix for the base fit, or None if no base snapshot is set.
		"""
		return None if self.base_snapshot is None else self.base_snapshot.weights

	@property
	def ref_predictions(self) -> dict[str, NDArray[np.float64]]:
		"""Return predictions for every restricted reference model.

		Returns: Mapping from cause name to the corresponding reduced-model outputs.
		"""
		return {name: snap.predictions for name, snap in self.reference_snapshots.items()}

	@property
	def ref_weights(self) -> dict[str, NDArray[np.float64]]:
		"""Return weight matrices for every restricted reference model.

		Returns: Mapping from cause name to reduced-model weights for non-None
		entries.
		"""
		return {name: snap.weights for name, snap in self.reference_snapshots.items() if snap.weights is not None}

	@property
	def n_samples(self) -> int:
		"""Return the number of sample rows in the observed target data.

		Returns: Number of rows in the truth matrix, or zero when no data is stored.
		"""
		return int(self.y_true.shape[0]) if self.y_true.size else 0

	@property
	def p_value(self) -> pd.DataFrame:
		"""Return the p-value matrix for the evaluated Granger tests.

		Returns: DataFrame containing p-values indexed by effects and causes.
		"""
		return self.evaluate("p_value", restore_removed=True).data

	@property
	def f_test(self) -> pd.DataFrame:
		"""Return the F-statistic matrix for evaluated Granger tests.

		Returns: DataFrame containing F-statistics indexed by effects and causes.
		"""
		return self.evaluate("f_statistic", restore_removed=True).data

	@property
	def wald_test(self) -> pd.DataFrame:
		"""Return the Wald-test matrix for evaluated Granger tests.

		Returns: DataFrame containing Wald statistics indexed by effects and causes.
		"""
		return self.evaluate("wald_test", restore_removed=True).data

	@property
	def likelihood_ratio(self) -> pd.DataFrame:
		"""Return the likelihood-ratio matrix for evaluated Granger tests.

		Returns: DataFrame containing likelihood-ratio statistics indexed by effects
		and causes.
		"""
		return self.evaluate("likelihood_ratio", restore_removed=True).data

	@property
	def sign(self) -> pd.DataFrame:
		"""Return coefficient-derived signs with removed variables restored."""
		return self.evaluate("sign", restore_removed=True).data

	def set_dof_method(self, method: DofMethod) -> None:
		"""
		Select the convention used to count residual degrees of freedom.

		Args:
			method: Either ``"structural"`` or ``"active_support"``.

		Raises:
			ValueError: If the method name is unsupported.
		"""
		if method not in {"structural", "active_support"}:
			raise ValueError("dof method must be 'structural' or 'active_support'")
		self.dof_method = method

	def _weights_as_feature_output(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
		"""
		Normalize a coefficient matrix into effects-by-feature order.

		Args:
			weights: Raw coefficient matrix for the fitted model.

		Returns:
			Weight matrix arranged with one row per effect and one column per
			feature slot.

		Raises:
			ValueError: If the input is not a 2D array.
		"""
		weights_arr = np.asarray(weights, dtype=np.float64)
		if weights_arr.ndim != 2:
			raise ValueError(f"Expected 2D weights, got shape {weights_arr.shape}")
		if weights_arr.shape[0] == len(self.effects) and weights_arr.shape[1] != len(self.effects):
			return weights_arr
		if weights_arr.shape[1] == len(self.effects):
			return weights_arr.T
		return weights_arr

	def _effect_cause_matrix(self, values: NDArray[np.float64], *, name: str) -> CausalityMatrix:
		"""
		Wrap a numeric table as an effects-by-causes matrix.

		Args:
			values: Metric values to place into the matrix body.
			name: Human-readable label assigned to the resulting matrix.

		Returns:
			Data structure containing an effects-by-causes DataFrame and
			associated metadata.
		"""
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
		"""
		Attach preprocessing metadata used to restore removed variables.

		Args:
			context: Context object storing the prepared-data pipeline and the
				removed-variable restoration metadata.
		"""
		self.preparation_context = context

	def set_base_snapshot(
			self,
			base_predictions: NDArray[np.float64],
			base_weights: NDArray[np.float64],
			metadata: Optional[dict[str, Any]] = None,
		) -> None:
			"""
			Store the unrestricted base-model snapshot.

			Args:
				base_predictions: Predicted values from the unrestricted fit.
				base_weights: Coefficient matrix for the unrestricted fit.
				metadata: Optional metadata describing the fitted model state.
			"""
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
			"""
			Store the reduced model that omits a single cause variable.

			Args:
				cause: Name of the omitted cause variable.
				reference_predictions: Predicted values from the reduced model.
				reference_weights: Coefficient matrix of the reduced model.
				metadata: Optional metadata describing the restricted fit.
			"""
			self.reference_snapshots[cause] = ModelSnapshot(
				predictions=ensure_2d(reference_predictions),
				weights=np.asarray(reference_weights, dtype=np.float64),
				metadata=metadata or {},
			)
			self._rss_cache.pop(f"reference:{cause}", None)
			self._metric_cache.clear()

	def n_restrictions_for(self, cause: str) -> NDArray[np.int_]:
		"""
		Count which coefficients are active for one cause across all effects.

		Args:
			cause: Name of the cause whose coefficient block is being counted.

		Returns:
			1D array containing the number of active restrictions for each effect.

		Raises:
			ValueError: If the cause name is not part of the configured causal set.
		"""
		if cause not in self.causes:
			raise ValueError(f"Unknown tested cause: {cause!r}")

		if cause not in self.cause_slices:
			raise ValueError(
				f"Missing feature slice for tested cause: {cause!r}"
			)

		if self.mask.size == 0:
			return np.zeros(len(self.effects), dtype=np.int_)

		if self.mask.ndim != 2 or self.mask.shape[0] != len(self.effects):
			raise ValueError(
				"Expected mask with shape "
				f"(n_effects, n_features); got {self.mask.shape}"
			)

		block = self.cause_slices[cause]
		if not isinstance(block, slice):
			raise ValueError(
				f"Expected slice for cause {cause!r}; got {type(block).__name__}"
			)

		if (
			block.start is None
			or block.stop is None
			or block.step not in (None, 1)
			or not 0 <= block.start <= block.stop <= self.mask.shape[1]
		):
			raise ValueError(
				f"Invalid feature slice for cause {cause!r}: {block!r}; "
				f"mask has {self.mask.shape[1]} feature columns"
			)

		mask_block = self.mask[:, block]

		if (
			self.dof_method == "active_support"
			and self.base_snapshot is not None
			and self.base_snapshot.weights is not None
		):
			weights = self._weights_as_feature_output(
				self.base_snapshot.weights
			)

			if weights.shape != self.mask.shape:
				raise ValueError(
					f"Weight shape {weights.shape} does not match "
					f"mask shape {self.mask.shape}"
				)

			mask_block = mask_block & (np.abs(weights[:, block]) > 0.0)

		return np.asarray(mask_block.sum(axis=1), dtype=np.int_)
	
	def n_unrestricted_parameters(self) -> int:
		"""
		Calculate the model-width proxy used for residual degree-of-freedom
		calculations.

		Returns:
			Number of unrestricted parameters, falling back to the stored
			explanatory matrix width when no base snapshot is present.
		"""
		if self.base_snapshot is not None and self.base_snapshot.weights is not None:
			weights = self._weights_as_feature_output(self.base_snapshot.weights)
			return int(max(weights.shape[1], 1))
		if self.X_true.size:
			return int(max(self.X_true.shape[1], 1))
		return 1

	def base_rss_per_output(self) -> NDArray[np.float64]:
		"""
		Compute the residual sum of squares for the unrestricted base model.

		Returns:
			1D array of RSS values, one per output variable.

		Raises:
			ValueError: If no base snapshot has been registered.
		"""
		if self.base_snapshot is None:
			raise ValueError("Base snapshot is not available")
		if "base" not in self._rss_cache:
			self._rss_cache["base"] = residual_sum_of_squares(
				self.y_true,
				self.base_snapshot.predictions,
			)
		return self._rss_cache["base"]

	def reference_rss_per_output(self, cause: str) -> NDArray[np.float64]:
		"""
		Compute the residual sum of squares for the restricted model omitting
		one cause.

		Args:
			cause: Name of the omitted cause variable.

		Returns:
			RSS values for each output in the reference model.

		Raises:
			ValueError: If the cause does not have a stored reference snapshot.
		"""
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
		"""
		Build a metric matrix indexed by effects and causes.

		Args:
			values: Numeric values to populate the matrix with.
			name: Identifier used by the returned CausalityMatrix.
			metadata: Optional metadata attached to the output matrix.

		Returns:
			A CausalityMatrix with an effects-by-causes DataFrame payload.
		"""
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
			"""
			Resolve and evaluate a Granger-causality metric.

			Args:
				metric: Metric name or callable to execute.
				restore_removed: If true, rebuild removed-variable entries before
					returning the metric.
				cache: Whether to reuse a previously computed result.
				force: Ignore cached results when true.
				**kwargs: Extra arguments forwarded to the metric implementation.

			Returns:
				The metric matrix with optional restoration applied.
			"""
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
						effects=self.preparation_context.requested_roles.effects,
						causes=self.preparation_context.requested_roles.tested_causes,
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
			"""
			Return a thresholded effect-cause table for the selected metric.

			Args:
				threshold: Significance threshold used to convert a continuous
					statistic into binary evidence.
				with_sign: Whether to multiply the binary result by the net causal
					sign matrix.
				metric: Metric to evaluate before thresholding.

			Returns:
				DataFrame with 1/0 indicator values after thresholding.
			"""
			metric_matrix = self.evaluate(metric, restore_removed=True)
			binary = (metric_matrix.data < threshold).astype(int)

			if not with_sign:
				return binary

			signs = self.sign

			if (
				not binary.index.equals(signs.index)
				or not binary.columns.equals(signs.columns)
			):
				raise ValueError(
					"Restored metric and sign matrices must have identical "
					"effect and cause axes."
				)

			return binary.mul(signs)

	def _sign_matrix(self) -> pd.DataFrame:
		"""
		Summarize the net direction of each causal effect from base weights.
		Unused variables are not restored to the matrix before returning.
		Returns:
			DataFrame containing a sign value for each effect-cause pair.
		"""
		return self.evaluate("sign", restore_removed=False).data

	
	def __repr__(self) -> str:
		"""Return a compact textual summary of the result container."""
		return (
			f"GrangerAnalysisResults(effects={self.effects}, causes={self.causes}, "
			f"dof_method={self.dof_method!r})"
		)
		