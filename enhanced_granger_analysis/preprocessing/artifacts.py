from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from ..core.data.analysis_roles import VariableRoles
from ..core.data.feature_layout import FeatureLayout
from ..core.data.prepared import PreparedData
from ..core.data.provenance import LagSegmentInfo, RemovedVariable
from ..core.data.split import DataSplit
from ..core.data.variables import VariableDescription
from ..core.exceptions import DataValidationError


@dataclass
class _PreparationState:
    """Temporary workspace of the existing staged preparer; not a result."""

    stationarity_enforcer: Optional[Any] = None
    lag_engine: Optional[Any] = None
    x_scaler: Any = None
    y_scaler: Any = None
    X_scaled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    y_scaled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    X_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    y_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    base_mask: Optional[np.ndarray] = None
    requested_roles: VariableRoles = field(
        default_factory=lambda: VariableRoles((), (), ())
    )
    effective_roles: VariableRoles = field(
        default_factory=lambda: VariableRoles((), (), ())
    )
    cause_slices: dict[str, slice] = field(default_factory=dict)
    feature_lags: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    removed_variables: dict[str, RemovedVariable] = field(default_factory=dict)
    variable_descriptions: dict[str, VariableDescription] = field(default_factory=dict)
    sample_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    lag_fitted: Optional[Any] = None
    split: Optional[DataSplit] = None
    segments: tuple[LagSegmentInfo, ...] = ()
    original_row_counts: tuple[int, ...] = ()

    def resolved_split(self) -> DataSplit:
        n_rows = self.X_raw.shape[0]
        if self.y_raw.shape[0] != n_rows:
            raise DataValidationError("X_raw and y_raw have different row counts.")
        if self.split is None:
            return DataSplit.all_training(n_rows)
        return self.split.validate(n_rows)

    def finalize_split(self) -> DataSplit:
        split = self.resolved_split()
        self.split = split
        return split

    def indices_for(self, part: str, *, require_nonempty: bool = False) -> np.ndarray:
        return self.resolved_split().indices_for(part, require_nonempty=require_nonempty)

    def split_snapshot(self) -> dict:
        return self.resolved_split().to_dict(n_rows=self.X_raw.shape[0])

    def as_prepared_data(self) -> PreparedData:
        """Publish a result without exposing the live workspace or a fit API."""
        return PreparedData(
            X_raw=self.X_raw,
            y_raw=self.y_raw,
            X_scaled=self.X_scaled,
            y_scaled=self.y_scaled,
            feature_layout=FeatureLayout(
                self.effective_roles.causes, self.cause_slices, self.feature_lags,
            ),
            requested_roles=self.requested_roles,
            effective_roles=self.effective_roles,
            base_mask=self.base_mask,
            split=self.resolved_split(),
            sample_indices=self.sample_indices,
            target_inverse=deepcopy(self.y_scaler).inverse_transform,
            segments=tuple(
                replace(segment, original_row_count=self.original_row_counts[segment.dataset_index])
                for segment in self.segments
            ),
            removed_variables=self.removed_variables,
            variable_descriptions=self.variable_descriptions,
        )
