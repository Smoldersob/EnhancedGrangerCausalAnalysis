from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.data.analysis_roles import VariableRoles
from ..core.data.provenance import RemovedVariable
from ..core.exceptions import DataValidationError
from ..core.lags.specification import LagSpec
from ..preprocessing.splitting.base import DataSplit


@dataclass
class PreparedData:
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
    sample_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    lag_fitted: Optional[Any] = None
    split: Optional[DataSplit] = None

    @property
    def causes(self) -> tuple[str, ...]:
        return self.effective_roles.causes

    @property
    def effects(self) -> tuple[str, ...]:
        return self.effective_roles.effects

    @property
    def tested_causes(self) -> tuple[str, ...]:
        return self.effective_roles.tested_causes

    @property
    def tested_cause_slices(self) -> dict[str, slice]:
        return {
            name: self.cause_slices[name]
            for name in self.effective_roles.tested_causes
        }

    @property
    def predictor_slices(self) -> dict[str, slice]:
        return dict(self.cause_slices)

    @classmethod
    def empty(cls) -> "PreparedData":
        return cls(
            X_scaled=np.empty((0, 0), dtype=float),
            y_scaled=np.empty((0, 0), dtype=float),
            X_raw=np.empty((0, 0), dtype=float),
            y_raw=np.empty((0, 0), dtype=float),
            requested_roles=VariableRoles((), (), ()),
            effective_roles=VariableRoles((), (), ()),
        )

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
        return self.resolved_split().indices_for(
            part, require_nonempty=require_nonempty
        )

    def arrays_for(
        self,
        part: str,
        *,
        scaled: bool = True,
        require_nonempty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self.X_scaled if scaled else self.X_raw
        y = self.y_scaled if scaled else self.y_raw
        n_rows = self.X_raw.shape[0]
        if X.ndim != 2 or y.ndim != 2 or X.shape[0] != n_rows or y.shape[0] != n_rows:
            raise DataValidationError("Prepared matrices are not aligned.")
        indices = self.indices_for(part, require_nonempty=require_nonempty)
        return X[indices], y[indices]

    def split_snapshot(self) -> dict:
        return self.resolved_split().to_dict(n_rows=self.X_raw.shape[0])
