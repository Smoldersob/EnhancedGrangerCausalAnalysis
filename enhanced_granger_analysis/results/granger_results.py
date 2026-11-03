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

DofMethod = Literal["structural", "active_support", "rank"]


@dataclass
class ModelSnapshot:
	"""
	Snapshot of one fitted model run.
	Stores predictions, optional coefficient weights, and arbitrary
	metadata associated with a single unrestricted or restricted model fit.
	"""

	predictions: NDArray[np.float64]
	weights: NDArray[np.float64] | None = None
	active_slope_mask: NDArray[np.bool_] | None = None
	bias_enabled: bool = True
	snapshot_id: str | None = None
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
	def stationarity_enforcer(self):
		"""Return the fitted stationarity enforcer, or None if disabled."""
		return self.prepared_data.stationarity_enforcer
	
	@property
	def lag_engine(self):
		"""Return the lag-generation engine used by the analysis.

		Returns: Lag-engine object producing time-lagged regression features.
		"""
		return self.prepared_data.lag_engine

	@property
	def split(self):
		"""Return the active data split, or None when splitting is disabled.

		Returns: Assignment of prepared rows to named parts.
		"""
		return self.prepared_data.split

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
		self.active_support_tolerance = 1e-12
		self.rank_rcond = None
		self._rank_cache = {}
		self.base_snapshot: ModelSnapshot | None = None
		self.reference_snapshots: dict[str, ModelSnapshot] = {}
		self.restricted_snapshots: dict[str, ModelSnapshot] = self.reference_snapshots
		self.unrestricted_snapshots: dict[str, ModelSnapshot] = {}
		self.base_model_role: Literal["unrestricted", "restricted"] | None = None
		self._metric_cache: dict[Any, CausalityMatrix] = {}
		self._rss_cache: dict[str, NDArray[np.float64]] = {}
		self.preparation_context: Optional[PreparationContext] = None
		self.model_search_result: Any = None
		self.effective_modeling: Any = None

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

	def set_dof_method(self, method, *, rank_rcond=None):
		"""Select structural, active-support, or numerical-rank counting."""
		if method not in {"structural", "active_support", "rank"}:
			raise ValueError("dof method must be structural, active_support, or rank")
		if rank_rcond is not None and (isinstance(rank_rcond, (bool, np.bool_))
				or not np.isfinite(rank_rcond) or not 0 <= rank_rcond < 1):
			raise ValueError("rank_rcond must be None or a finite number in [0, 1)")
		self.dof_method = method
		self.rank_rcond = None if rank_rcond is None else float(rank_rcond)
		self._metric_cache.clear()
		self._rank_cache.clear()

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
		expected = (len(self.effects), self.mask.shape[1])
		if weights_arr.shape != expected:
			raise ValueError(
				f"Expected normalized weights with shape {expected}, got {weights_arr.shape}"
			)
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
			active_slope_mask: NDArray[np.bool_] | None = None,
			bias_enabled: bool = True,
			role: Literal["unrestricted", "restricted"] | None = "unrestricted",
		) -> None:
			"""
			Store the unrestricted base-model snapshot.

			Args:
				base_predictions: Predicted values from the unrestricted fit.
				base_weights: Coefficient matrix for the unrestricted fit.
				metadata: Optional metadata describing the fitted model state.
			"""
			self.base_snapshot = ModelSnapshot(
				predictions=ensure_2d(np.array(base_predictions, dtype=np.float64, copy=True)),
				weights=np.array(base_weights, dtype=np.float64, copy=True),
				active_slope_mask=None if active_slope_mask is None else np.array(active_slope_mask, dtype=bool, copy=True),
				bias_enabled=bool(bias_enabled),
				metadata=dict(metadata or {}),
			)
			self.base_model_role = role
			if role == "unrestricted":
				self.unrestricted_snapshots = {cause: self.base_snapshot for cause in self.causes}
			elif role == "restricted":
				self.restricted_snapshots = {cause: self.base_snapshot for cause in self.causes}
			self.reference_snapshots = self.restricted_snapshots
			self._rss_cache.clear()
			self._metric_cache.clear()
			self._rank_cache.clear()

	def set_reference_snapshot(
			self,
			cause: str,
			reference_predictions: NDArray[np.float64],
			reference_weights: NDArray[np.float64],
			metadata: Optional[dict[str, Any]] = None,
			active_slope_mask: NDArray[np.bool_] | None = None,
			bias_enabled: bool = True,
		) -> None:
			"""
			Store the reduced model that omits a single cause variable.

			Args:
				cause: Name of the omitted cause variable.
				reference_predictions: Predicted values from the reduced model.
				reference_weights: Coefficient matrix of the reduced model.
				metadata: Optional metadata describing the restricted fit.
			"""
			snapshot = ModelSnapshot(
				predictions=ensure_2d(np.array(reference_predictions, dtype=np.float64, copy=True)),
				weights=np.array(reference_weights, dtype=np.float64, copy=True),
				active_slope_mask=None if active_slope_mask is None else np.array(active_slope_mask, dtype=bool, copy=True),
				bias_enabled=bool(bias_enabled),
				metadata=dict(metadata or {}),
			)
			self.reference_snapshots[cause] = snapshot
			self.restricted_snapshots = self.reference_snapshots
			self.restricted_snapshots[cause] = snapshot
			self._rss_cache.clear()
			self._metric_cache.clear()
			self._rank_cache.clear()

	def set_unrestricted_snapshot(
		self,
		cause: str,
		predictions: NDArray[np.float64],
		weights: NDArray[np.float64],
		metadata: Optional[dict[str, Any]] = None,
		active_slope_mask: NDArray[np.bool_] | None = None,
		bias_enabled: bool = True,
	) -> None:
		snapshot = ModelSnapshot(
			predictions=ensure_2d(np.array(predictions, dtype=np.float64, copy=True)),
			weights=np.array(weights, dtype=np.float64, copy=True),
			active_slope_mask=None if active_slope_mask is None else np.array(active_slope_mask, dtype=bool, copy=True),
			bias_enabled=bool(bias_enabled),
			metadata=dict(metadata or {}),
		)
		self.unrestricted_snapshots[cause] = snapshot
		self._rss_cache.clear()
		self._metric_cache.clear()
		self._rank_cache.clear()

	def set_restricted_snapshot(
		self,
		cause: str,
		predictions: NDArray[np.float64],
		weights: NDArray[np.float64],
		metadata: Optional[dict[str, Any]] = None,
		active_slope_mask: NDArray[np.bool_] | None = None,
		bias_enabled: bool = True,
	) -> None:
		self.set_reference_snapshot(
			cause,
			predictions,
			weights,
			metadata,
			active_slope_mask,
			bias_enabled,
		)

	def register_snapshot(
		self,
		cause: str,
		role: Literal["restricted", "unrestricted"],
		snapshot: ModelSnapshot,
	) -> None:
		"""Register an existing snapshot for one cause and analysis role."""
		if role == "restricted":
			self.restricted_snapshots[cause] = snapshot
			self.reference_snapshots = self.restricted_snapshots
		elif role == "unrestricted":
			self.unrestricted_snapshots[cause] = snapshot
		else:
			raise ValueError("role must be 'restricted' or 'unrestricted'")
		self._rss_cache.clear()
		self._metric_cache.clear()
		self._rank_cache.clear()

	def get_snapshot(
		self,
		cause: str,
		role: Literal["restricted", "unrestricted"],
	) -> ModelSnapshot:
		"""Retrieve the snapshot registered for a cause and role."""
		return self._snapshot_for(cause, role)

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
		if self.dof_method == "rank":
			return np.array([g["q"] for g in self._rank_geometry_for(cause)], dtype=np.int_)
		if cause not in self.causes:
			raise ValueError(f"Unknown tested cause: {cause!r}")

		if cause not in self.cause_slices:
			raise ValueError(
				f"Missing feature slice for tested cause: {cause!r}"
			)

		unrestricted = self._snapshot_for(cause, "unrestricted")
		restricted = self._snapshot_for(cause, "restricted")
		u_mask = self._require_snapshot_mask(unrestricted)
		r_mask = self._require_snapshot_mask(restricted)
		if u_mask.shape != r_mask.shape or u_mask.shape[0] != len(self.effects):
			raise ValueError("Restricted and unrestricted masks must have matching output/feature shapes.")
		if np.any(r_mask & ~u_mask):
			raise ValueError("Restricted active positions must be a subset of unrestricted positions.")

		block = self.cause_slices[cause]
		if not isinstance(block, slice):
			raise ValueError(
				f"Expected slice for cause {cause!r}; got {type(block).__name__}"
			)

		if (
			block.start is None
			or block.stop is None
			or block.step not in (None, 1)
			or not 0 <= block.start <= block.stop <= u_mask.shape[1]
		):
			raise ValueError(
				f"Invalid feature slice for cause {cause!r}: {block!r}; "
				f"mask has {u_mask.shape[1]} feature columns"
			)

		if unrestricted.bias_enabled != restricted.bias_enabled:
			raise ValueError("Compared snapshots must use matching intercept settings")
		tested = u_mask & ~r_mask
		if self.dof_method == "active_support":
			if unrestricted.weights is None:
				raise ValueError("Active-support counts require unrestricted weights.")
			weights = self._weights_as_feature_output(unrestricted.weights)
			if weights.shape != u_mask.shape:
				raise ValueError("Unrestricted weights and mask must have matching shapes.")
			tested = tested & (np.abs(weights) > self.active_support_tolerance)
		return np.asarray(tested.sum(axis=1), dtype=np.int_)
	
	def n_unrestricted_parameters(self, cause=None):
		"""Count enabled unrestricted slopes and bias per output."""
		if self.dof_method == "rank":
			if cause is None:
				if not self.causes:
					raise ValueError("A tested cause is required")
				cause = self.causes[0]
			return np.array([g["rank_u"] for g in self._rank_geometry_for(cause)], dtype=np.int_)
		if cause is None:
			if not self.causes:
				raise ValueError("A tested cause is required")
			cause = self.causes[0]
		snapshot = self._snapshot_for(cause, "unrestricted")
		mask = self._require_snapshot_mask(snapshot).copy()
		if self.dof_method == "active_support":
			if snapshot.weights is None:
				raise ValueError("Active-support counts require unrestricted weights")
			weights = self._weights_as_feature_output(snapshot.weights)
			mask &= np.abs(weights) > self.active_support_tolerance
		return mask.sum(axis=1).astype(np.int_) + int(snapshot.bias_enabled)

	def _rank_coordinates(self, snapshot):
		"""Return observations and predictions in the snapshot's model coordinates."""
		space = snapshot.metadata.get("coefficient_space")
		if space == "model":
			if self.preparation_context is None:
				raise ValueError("Rank evaluation requires the model preparation context")
			data = self.preparation_context.prepared_data
			x, y = data.X_scaled, data.y_scaled
			pred = snapshot.metadata.get("model_predictions")
			if pred is None:
				raise ValueError("Snapshot lacks model-space predictions; refit")
		elif space == "raw" or (space is None and self.preparation_context is None):
			x, y, pred = self.X_true, self.y_true, snapshot.predictions
		else:
			raise ValueError("Ambiguous snapshot coordinates; refit or declare raw coordinates")
		x, y, pred = [np.asarray(a, dtype=float) for a in (x, y, pred)]
		if x.ndim != 2 or y.ndim != 2 or pred.shape != y.shape:
			raise ValueError("Invalid rank-evaluation array dimensions")
		if x.shape[0] != self.n_samples or y.shape != self.y_true.shape:
			raise ValueError("Rank evaluation requires the same observations and output order")
		if not all(np.isfinite(a).all() for a in (x, y, pred)):
			raise ValueError("Rank-evaluation arrays must be finite")
		return x, y, pred


	def _rank_geometry_for(self, cause):
		"""Cache the nested design geometry per cause and snapshot pair."""
		from .statistics import nested_design_geometry
		if cause not in self.causes:
			raise ValueError(f"Unknown tested cause: {cause!r}")
		u, r = self.get_snapshot(cause, "unrestricted"), self.get_snapshot(cause, "restricted")
		key = (id(u), id(r), self.rank_rcond)
		if key in self._rank_cache:
			return self._rank_cache[key]
		um, rm = self._require_snapshot_mask(u), self._require_snapshot_mask(r)
		x, _, _ = self._rank_coordinates(u)
		expected = (len(self.effects), x.shape[1])
		if um.shape != expected or rm.shape != expected or np.any(rm & ~um):
			raise ValueError("Rank mode requires matching nested slope masks")
		if u.bias_enabled != r.bias_enabled:
			raise ValueError("Rank mode requires matching intercept settings")
		geometries = []
		for j in range(len(self.effects)):
			du, dr = x[:, um[j]], x[:, rm[j]]
			if u.bias_enabled:
				ones = np.ones((len(x), 1))
				du, dr = np.column_stack([ones, du]), np.column_stack([ones, dr])
			geometries.append(nested_design_geometry(du, dr, rcond=self.rank_rcond))
		self._rank_cache[key] = geometries
		return geometries


	def rank_wald_per_output(self, cause):
		"""Test identifiable added prediction directions, not arbitrary aliased coefficients.

		This rank-mode Wald uses fitted unrestricted means and classical OLS
		residual variance. It does not need a fitted restricted model.
		"""
		u = self.get_snapshot(cause, "unrestricted")
		_, y, pred = self._rank_coordinates(u)
		values = np.full(len(self.effects), np.nan)
		for j, g in enumerate(self._rank_geometry_for(cause)):
			if g["q"] == 0:
				continue
			df = self.n_samples - g["rank_u"]
			if df <= 0:
				raise ValueError("Rank-mode residual degrees of freedom must be positive")
			qu, qa = g["basis_u"], g["basis_added"]
			# Reject a changed effective model rather than silently mixing rank
			# truncation with a fit that used different prediction directions.
			projection_error = np.linalg.norm(pred[:, j] - qu @ (qu.T @ pred[:, j]))
			if projection_error > 1e-7 * max(1., np.linalg.norm(pred[:, j])):
				raise ValueError("Fitted U is outside the retained rank space; align fitting and rank tolerance")
			score = np.linalg.norm(qu.T @ (y[:, j] - pred[:, j]))
			if score > 1e-7 * max(1., np.linalg.norm(qu.T @ y[:, j])):
				raise ValueError("Rank-mode Wald requires an OLS-equivalent unrestricted fit")
			sigma2 = np.sum((y[:, j] - pred[:, j]) ** 2) / df
			numerator = np.sum((qa.T @ pred[:, j]) ** 2)
			if sigma2 > 0:
				values[j] = numerator / sigma2
			elif numerator > 0:
				values[j] = np.inf
		return values


	def rank_diagnostics(self):
		"""Return per-output design ranks and structural counts as a DataFrame."""
		records = []
		for cause in self.causes:
			u, r = self.get_snapshot(cause, "unrestricted"), self.get_snapshot(cause, "restricted")
			um, rm = self._require_snapshot_mask(u), self._require_snapshot_mask(r)
			for j, g in enumerate(self._rank_geometry_for(cause)):
				s = g["singular_u"]
				condition = float(s[0] / s[-1]) if s.size and s[-1] > 0 else np.inf
				records.append(dict(cause=cause, effect=self.effects[j],
					k_u_structural=int(um[j].sum()) + int(u.bias_enabled),
					k_u_rank=g["rank_u"], k_r_rank=g["rank_r"],
					q_structural=int(np.count_nonzero(um[j] & ~rm[j])), q_rank=g["q"],
					residual_df=self.n_samples-g["rank_u"], cutoff=g["cutoff"],
					smallest_singular=float(s[-1]) if s.size else 0., condition=condition,
					restricted_fit_performed=r.metadata.get("fit_performed")))
		return pd.DataFrame.from_records(records)

	def _snapshot_for(self, cause: str, role: Literal["restricted", "unrestricted"]) -> ModelSnapshot:
		mapping = self.restricted_snapshots if role == "restricted" else self.unrestricted_snapshots
		if cause not in mapping:
			raise ValueError(f"{role.title()} snapshot for cause '{cause}' is not available")
		return mapping[cause]

	@staticmethod
	def _require_snapshot_mask(snapshot: ModelSnapshot) -> NDArray[np.bool_]:
		if snapshot.active_slope_mask is None:
			raise ValueError("Snapshot is missing its active slope mask")
		return np.array(snapshot.active_slope_mask, dtype=bool, copy=False)

	def unrestricted_rss_per_output(self, cause: str) -> NDArray[np.float64]:
		snapshot = self._snapshot_for(cause, "unrestricted")
		key = f"unrestricted:{id(snapshot)}"
		if key not in self._rss_cache:
			self._rss_cache[key] = residual_sum_of_squares(self.y_true, snapshot.predictions)
		return self._rss_cache[key]

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
		snapshot = self._snapshot_for(cause, "restricted")
		cache_key = f"restricted:{id(snapshot)}"
		if cache_key not in self._rss_cache:
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
		