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
from .dof import (
    DofContext,
    DofMethod,
    DofResult,
    dof_name,
    evaluate_dof,
    rank_geometries,
    resolve_dof,
)

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
            X_true: Original explanatory data matrix used for raw-coordinate
                snapshot evaluation.
        """
        self.effects = list(effects or [])
        self.causes = list(causes or [])
        self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.empty((0, 0), dtype=bool)
        self.cause_slices = dict(cause_slices or {})
        self.y_true = ensure_2d(np.asarray(y_true, dtype=np.float64)) if y_true is not None else np.empty((0, 0), dtype=np.float64)
        self.X_true = ensure_2d(np.asarray(X_true, dtype=np.float64)) if X_true is not None else np.empty((0, 0), dtype=np.float64)
        self._dof_method: DofMethod = "structural"
        self.dof_params: dict[str, Any] = {}
        self.unavailable_policy = "zero"
        self._rank_cache = {}
        self.unrestricted_snapshots: dict[str, ModelSnapshot] = {}
        self.restricted_snapshots: dict[str, ModelSnapshot] = {}
        self._metric_cache: dict[Any, CausalityMatrix] = {}
        self._rss_cache: dict[str, NDArray[np.float64]] = {}
        self.preparation_context: Optional[PreparationContext] = None
        self.model_search_result: Any = None
        self.effective_modeling: Any = None

    @property
    def n_samples(self) -> int:
        """Return the number of sample rows in the observed target data.

        Returns: Number of rows in the truth matrix, or zero when no data is stored.
        """
        return int(self.y_true.shape[0]) if self.y_true.size else 0

    @property
    def f_p_value(self) -> pd.DataFrame:
        """Return F-test p-values with removed variables restored."""
        return self.evaluate(
            "f_statistic_p_value",
            restore_removed=True,
        ).data

    @property
    def wald_p_value(self) -> pd.DataFrame:
        """Return coefficient Wald p-values with removed variables restored."""
        return self.evaluate(
            "wald_test_p_value",
            restore_removed=True,
        ).data

    @property
    def likelihood_ratio_p_value(self) -> pd.DataFrame:
        """Return likelihood-ratio p-values with removed variables restored."""
        return self.evaluate(
            "likelihood_ratio_p_value",
            restore_removed=True,
        ).data

    @property
    def wald_test(self) -> pd.DataFrame:
        """Return the coefficient Wald statistic matrix with removed variables restored.

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
    def wald_projection(self) -> pd.DataFrame:
        """Return projection Wald statistics."""
        return self.evaluate(
            "wald_projection",
            restore_removed=True,
        ).data

    @property
    def wald_projection_p_value(self) -> pd.DataFrame:
        """Return projection Wald p-values."""
        return self.evaluate(
            "wald_projection_p_value",
            restore_removed=True,
        ).data

    @property
    def sign(self) -> pd.DataFrame:
        """Return coefficient-derived signs with removed variables restored."""
        return self.evaluate("sign", restore_removed=True).data

    ### Preparation context and removed-variable restoration

    def set_preparation_context(self, context: PreparationContext) -> None:
        """
        Attach preprocessing metadata used to restore removed variables.

        Args:
            context: Context object storing the prepared-data pipeline and the
                removed-variable restoration metadata.
        """
        self.preparation_context = context

    ### Saving and retrieving snapshots for later evaluation

    def _invalidate_snapshot_caches(self) -> None:
        """Invalidate values derived from model snapshots."""
        self._rss_cache.clear()
        self._metric_cache.clear()
        self._rank_cache.clear()

    def clear_snapshots(self) -> None:
        """Remove model snapshots without changing analysis configuration."""
        self.unrestricted_snapshots.clear()
        self.restricted_snapshots.clear()
        self._invalidate_snapshot_caches()

    def _snapshots_for_role(
        self,
        role: Literal["restricted", "unrestricted"],
    ) -> dict[str, ModelSnapshot]:
        """Return the snapshot mapping for an explicit model role."""
        if role == "unrestricted":
            return self.unrestricted_snapshots
        if role == "restricted":
            return self.restricted_snapshots

        raise ValueError(
            "Snapshot role must be 'restricted' or 'unrestricted'"
        )

    def register_snapshots(
        self,
        causes: Iterable[str],
        role: Literal["restricted", "unrestricted"],
        snapshot: ModelSnapshot,
    ) -> None:
        """Register one snapshot for multiple causes without copying it.

        Args:
            causes: Causes whose comparisons use this snapshot.
            role: Statistical role of the snapshot.
            snapshot: Existing snapshot to register.
        """
        snapshots = self._snapshots_for_role(role)

        if not isinstance(snapshot, ModelSnapshot):
            raise TypeError("snapshot must be a ModelSnapshot")

        if isinstance(causes, str):
            raise TypeError(
                "causes must be an iterable of names, not a single string"
            )

        selected_causes = tuple(causes)
        unknown = [
            cause
            for cause in selected_causes
            if cause not in self.causes
        ]

        if unknown:
            raise ValueError(f"Unknown tested causes: {unknown!r}")

        for cause in selected_causes:
            snapshots[cause] = snapshot

        self._invalidate_snapshot_caches()

    def register_snapshot(
        self,
        cause: str,
        role: Literal["restricted", "unrestricted"],
        snapshot: ModelSnapshot,
    ) -> None:
        """Register an existing snapshot for one cause and model role."""
        self.register_snapshots(
            causes=(cause,),
            role=role,
            snapshot=snapshot,
        )

    def get_snapshot(
        self,
        cause: str,
        role: Literal["restricted", "unrestricted"],
    ) -> ModelSnapshot:
        """Return the snapshot registered for a cause and model role."""
        if cause not in self.causes:
            raise ValueError(f"Unknown tested cause: {cause!r}")

        snapshots = self._snapshots_for_role(role)

        try:
            return snapshots[cause]
        except KeyError as exc:
            raise ValueError(
                f"{role.title()} snapshot for cause {cause!r} "
                "is not available"
            ) from exc

    #### Residual sum-of-squares evaluation methods

    def rss_per_output(
        self,
        cause: str,
        role: Literal["restricted", "unrestricted"],
    ) -> NDArray[np.float64]:
        """Return residual sums of squares for a registered snapshot.

        Args:
            cause: Cause identifying the model comparison.
            role: Statistical role of the requested snapshot.
        """
        snapshot = self.get_snapshot(cause, role)
        key = f"snapshot:{id(snapshot)}"

        if key not in self._rss_cache:
            self._rss_cache[key] = residual_sum_of_squares(
                self.y_true,
                snapshot.predictions,
            )

        return self._rss_cache[key].copy()

    ### DOF strategy selection and evaluation

    @property
    def dof_method(self) -> DofMethod:
        """Return the selected name or ready DOF strategy."""
        return self._dof_method

    @dof_method.setter
    def dof_method(self, method: DofMethod) -> None:
        """Select a DOF strategy and invalidate dependent values."""
        resolve_dof(method)

        self._dof_method = (
            dof_name(method) if isinstance(method, str) else method
        )

        if hasattr(self, "_metric_cache"):
            self._metric_cache.clear()

        if hasattr(self, "_rank_cache"):
            self._rank_cache.clear()

    def set_dof_method(
        self,
        method: DofMethod,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Select a DOF strategy and pass parameters into its context.

        Args:
            method: Registered strategy name or ready callable.
            params: Arbitrary strategy options forwarded to
                ``DofContext.params``. Built-in strategies read
                ``rank_rcond`` and ``active_support_tolerance`` from it.
        """
        resolve_dof(method)
        self.dof_params = dict(params) if params else {}
        self.dof_method = method

    def _dof_context_for(self, cause: str) -> DofContext:
        """Prepare snapshot inputs without selecting a DOF algorithm."""
        cause = self._resolve_tested_cause(cause)
        unrestricted = self.get_snapshot(cause, "unrestricted")
        restricted = self.get_snapshot(cause, "restricted")
        u_mask, r_mask = self._structural_masks_for(cause)

        weights = None
        if unrestricted.weights is not None:
            weights = np.array(
                self._weights_as_feature_output(unrestricted.weights),
                dtype=float,
                copy=True,
            )

        return DofContext(
            n_samples=self.n_samples,
            effects=tuple(self.effects),
            cause=cause,
            unrestricted_mask=u_mask,
            restricted_mask=r_mask,
            bias_enabled=unrestricted.bias_enabled,
            unrestricted_weights=weights,
            load_features=lambda: self._snapshot_coordinates(
                unrestricted
            )[0],
            params=dict(self.dof_params),
            unrestricted_metadata=dict(unrestricted.metadata),
            restricted_metadata=dict(restricted.metadata),
        )

    def dof_for(self, cause: str) -> DofResult:
        """Evaluate DOF for one cause in effective output order."""
        return evaluate_dof(
            self.dof_method,
            self._dof_context_for(cause),
        )

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

    def _resolve_tested_cause(self, cause: str | None) -> str:
        """Resolve a cause name, using the first cause when omitted."""
        if cause is None:
            if not self.causes:
                raise ValueError("A tested cause is required")
            cause = self.causes[0]

        if cause not in self.causes:
            raise ValueError(f"Unknown tested cause: {cause!r}")
        return cause

    def _structural_masks_for(
        self,
        cause: str,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Return validated unrestricted and restricted slope masks.

        Args:
            cause: Cause identifying the compared model snapshots.
        """
        unrestricted = self.get_snapshot(cause, "unrestricted")
        restricted = self.get_snapshot(cause, "restricted")

        u_mask = np.array(
            self._require_snapshot_mask(unrestricted),
            dtype=bool,
            copy=True,
        )
        r_mask = np.array(
            self._require_snapshot_mask(restricted),
            dtype=bool,
            copy=True,
        )

        expected = (len(self.effects), self.mask.shape[1])

        if u_mask.shape != expected or r_mask.shape != expected:
            raise ValueError(
                f"Snapshot masks must have shape {expected}; "
                f"got unrestricted={u_mask.shape}, restricted={r_mask.shape}"
            )

        if np.any(r_mask & ~u_mask):
            raise ValueError(
                "Restricted slope positions must be a subset "
                "of unrestricted positions"
            )

        if unrestricted.bias_enabled != restricted.bias_enabled:
            raise ValueError(
                "Compared snapshots must use matching intercept settings"
            )

        return u_mask, r_mask

    def _snapshot_coordinates(
        self,
        snapshot: ModelSnapshot,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return features, targets, and predictions in snapshot coordinates.

        Args:
            snapshot: Snapshot declaring raw or model coefficient coordinates.
        """
        space = snapshot.metadata.get("coefficient_space")

        if space == "model":
            if self.preparation_context is None:
                raise ValueError(
                    "Model-space evaluation requires preparation context"
                )

            prepared = self.preparation_context.prepared_data
            x = prepared.X_scaled
            y = prepared.y_scaled
            predictions = snapshot.metadata.get("model_predictions")

            if predictions is None:
                raise ValueError(
                    "Snapshot lacks model-space predictions; refit the analysis"
                )

        elif space == "raw" or (
            space is None and self.preparation_context is None
        ):
            x = self.X_true
            y = self.y_true
            predictions = snapshot.predictions

        else:
            raise ValueError(
                "Ambiguous snapshot coordinates; "
                "refit or declare raw coordinates"
            )

        x, y, predictions = (
            np.asarray(value, dtype=np.float64)
            for value in (x, y, predictions)
        )

        if (
            x.ndim != 2
            or y.ndim != 2
            or predictions.shape != y.shape
        ):
            raise ValueError("Invalid snapshot-coordinate array dimensions")

        if (
            x.shape[0] != self.n_samples
            or y.shape != self.y_true.shape
            or x.shape[1] != self.mask.shape[1]
        ):
            raise ValueError(
                "Snapshot evaluation requires matching observations, "
                "features, and output order"
            )

        if not all(
            np.isfinite(value).all()
            for value in (x, y, predictions)
        ):
            raise ValueError("Snapshot-coordinate arrays must be finite")

        return x, y, predictions

    def _rank_geometry_for(self, cause: str):
        """Return cached geometry independently of the selected DOF method."""
        cause = self._resolve_tested_cause(cause)
        unrestricted = self.get_snapshot(cause, "unrestricted")
        restricted = self.get_snapshot(cause, "restricted")

        params_key = tuple(
            sorted(
                (key, repr(value))
                for key, value in self.dof_params.items()
            )
        )
        key = (
            id(unrestricted),
            id(restricted),
            params_key,
        )

        if key not in self._rank_cache:
            self._rank_cache[key] = rank_geometries(
                self._dof_context_for(cause)
            )

        return self._rank_cache[key]

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
                    restricted_fit_performed=r.metadata.get("fit_performed"),
                    nesting_error=g["nesting_error"],
                    nesting_tolerance=g["nesting_tolerance"],
                    is_nested=g["is_nested"],
                    added_basis_valid=g["added_basis_valid"],
                    geometry_status=g["geometry_status"]))
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _require_snapshot_mask(snapshot: ModelSnapshot) -> NDArray[np.bool_]:
        if snapshot.active_slope_mask is None:
            raise ValueError("Snapshot is missing its active slope mask")
        return np.array(snapshot.active_slope_mask, dtype=bool, copy=False)

    ### Helper for building metric matrices in metric.py

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

    ### Metric evaluation and result summarization

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
        metric="f_test_p_value",
        *,
        unavailable_policy: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a thresholded effect-cause table for the selected metric.

        Args:
            threshold: Significance threshold used to convert a continuous
                statistic into binary evidence.
            with_sign: Whether to multiply the binary result by the net causal
                sign matrix.
            metric: Metric to evaluate before thresholding.
            unavailable_policy: Handling of unavailable tests; defaults to the
                container-level ``unavailable_policy``. ``zero`` maps missing
                decisions to 0, ``preserve`` keeps ``pd.NA``, and ``raise``
                fails when any test is unavailable.

        Returns:
            DataFrame with 1/0 indicator values after thresholding.
        """
        if (
            isinstance(threshold, (bool, np.bool_))
            or not np.isscalar(threshold)
            or not np.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be a finite number in [0, 1]")

        policy = (
            self.unavailable_policy
            if unavailable_policy is None
            else unavailable_policy
        )

        if policy not in {"zero", "preserve", "raise"}:
            raise ValueError(
                "unavailable_policy must be zero, preserve, or raise"
            )

        matrix = self.evaluate(metric, restore_removed=True)
        p_values = matrix.data

        invalid = (
            p_values.notna()
            & (
                ~np.isfinite(p_values)
                | (p_values < 0)
                | (p_values > 1)
            )
        )

        if invalid.to_numpy().any():
            raise ValueError(
                "result() requires p-values in [0, 1] or missing values"
            )

        missing = p_values.isna()

        if policy == "raise" and missing.to_numpy().any():
            raise ValueError("Some hypothesis tests are unavailable")

        binary = (p_values < threshold).astype("Int8")
        binary = binary.mask(missing, pd.NA)

        if with_sign:
            signs = self.sign

            if (
                not binary.index.equals(signs.index)
                or not binary.columns.equals(signs.columns)
            ):
                raise ValueError(
                    "Restored p-values and signs must have matching axes"
                )

            # Zero decisions remain zero even when the sign is unavailable.
            signed = binary.astype("Float64").mul(
                signs.astype("Float64")
            )
            binary = signed.mask(binary.eq(0).fillna(False), 0.0)

        if policy == "zero":
            binary = binary.fillna(0)
            if with_sign:
                return binary.astype(float)
            return binary.astype(int)

        return binary

    def __repr__(self) -> str:
        """Return a compact textual summary of the result container."""
        return (
            f"GrangerAnalysisResults(effects={self.effects}, causes={self.causes}, "
            f"dof_method={self.dof_method!r}, dof_params={self.dof_params!r})"
        )
