"""Final matrices and their coordinate contracts, independent of preparation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ..exceptions import DataValidationError
from ..lags.specification import LagSpec
from .analysis_roles import VariableRoles
from .feature_layout import FeatureLayout, binary_mask, validate_names
from .provenance import LagSegmentInfo, RemovedVariable
from .split import DataSplit
from .variables import VariableDescription


class TargetInverseTransform(Protocol):
    """A callable that undoes only target scaling, without fitting."""

    def __call__(self, values: np.ndarray, /) -> np.ndarray: ...


def identity_target_inverse(values: np.ndarray, /) -> np.ndarray:
    """Explicit inverse for unscaled targets."""
    return values


@dataclass(frozen=True, eq=False)
class PreparedData:
    """Prepared result; both raw and scaled matrices use the same rows/columns.

    Raw here means before scaling, not necessarily before stationarization.
    An empty ``segments`` tuple means source labels were not supplied.
    """

    X_raw: np.ndarray
    y_raw: np.ndarray
    X_scaled: np.ndarray
    y_scaled: np.ndarray
    feature_layout: FeatureLayout
    requested_roles: VariableRoles
    effective_roles: VariableRoles
    base_mask: np.ndarray
    split: DataSplit
    sample_indices: np.ndarray
    target_inverse: TargetInverseTransform
    segments: tuple[LagSegmentInfo, ...] = ()
    variable_descriptions: Mapping[str, VariableDescription] = field(default_factory=dict)
    removed_variables: Mapping[str, RemovedVariable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.feature_layout, FeatureLayout):
            raise TypeError("feature_layout must be a FeatureLayout.")
        if not isinstance(self.requested_roles, VariableRoles) or not isinstance(
            self.effective_roles, VariableRoles
        ):
            raise TypeError("requested_roles and effective_roles must be VariableRoles.")
        for roles in (self.requested_roles, self.effective_roles):
            validate_names(roles.causes, "causes")
            validate_names(roles.effects, "effects")
            validate_names(roles.tested_causes, "tested_causes")
            if not set(roles.tested_causes).issubset(roles.causes):
                raise DataValidationError("tested_causes must be a subset of causes.")
        if self.causes != self.feature_layout.causes or not self.effects:
            raise DataValidationError("Effective roles must match the feature layout and define outputs.")
        matrices = {}
        for name in ("X_raw", "y_raw", "X_scaled", "y_scaled"):
            matrix = np.asarray(getattr(self, name))
            width = len(self.feature_lags) if name.startswith("X") else len(self.effects)
            if matrix.ndim != 2 or matrix.shape[1] != width:
                raise DataValidationError(f"{name} must have {width} columns.")
            matrices[name] = matrix
        n_rows = matrices["X_raw"].shape[0]
        if any(matrix.shape[0] != n_rows for matrix in matrices.values()):
            raise DataValidationError("Prepared matrices are not aligned.")
        for name, matrix in matrices.items():
            array = matrix.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

        mask = binary_mask(self.base_mask, (len(self.effects), len(self.feature_lags)))
        if not mask.any(axis=0).all():
            raise DataValidationError("Every prepared feature must be active for at least one output.")
        group_masks = {}
        for column, feature in enumerate(self.feature_layout.features):
            key = (feature.source, feature.lag)
            active = mask[:, column]
            if feature.lag == 0 and feature.source in self.effects and active[self.effects.index(feature.source)]:
                raise DataValidationError("The base mask cannot enable a self lag 0.")
            if key in group_masks and not np.array_equal(group_masks[key], active):
                raise DataValidationError("Representation components must share a mask for the same source and lag.")
            group_masks[key] = active
        object.__setattr__(self, "base_mask", mask)

        if not isinstance(self.split, DataSplit):
            raise TypeError("split must be a DataSplit; use DataSplit.all_training when unsplit.")
        self.split.validate(n_rows)
        if not callable(self.target_inverse):
            raise TypeError("target_inverse must be a callable target-scale inverse.")

        indices = np.asarray(self.sample_indices)
        if indices.shape != (n_rows,) or indices.dtype.kind not in "iu":
            raise DataValidationError("sample_indices must contain one integer per prepared row.")
        if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise DataValidationError("sample_indices must be non-negative and strictly increasing.")
        segments = tuple(self.segments)
        next_row = 0
        for segment in segments:
            if not isinstance(segment, LagSegmentInfo):
                raise TypeError("segments must contain LagSegmentInfo objects.")
            if segment.prepared_row_slice.start != next_row:
                raise DataValidationError("segments must cover contiguous pre-sampling positions in order.")
            next_row = segment.prepared_row_slice.stop
        if segments and indices.size and indices[-1] >= next_row:
            raise DataValidationError("sample_indices address rows outside the source segments.")
        indices = indices.copy()
        indices.setflags(write=False)
        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "segments", segments)

        for name, record_type in (
            ("variable_descriptions", VariableDescription), ("removed_variables", RemovedVariable),
        ):
            records = getattr(self, name)
            if not isinstance(records, Mapping):
                raise TypeError(f"{name} must be a mapping.")
            if any(not isinstance(record, record_type) or key != record.name for key, record in records.items()):
                raise DataValidationError(f"{name} must map signal names to corresponding {record_type.__name__} objects.")
            object.__setattr__(self, name, MappingProxyType(dict(records)))

    @property
    def causes(self) -> tuple[str, ...]:
        return self.feature_layout.causes

    @property
    def effects(self) -> tuple[str, ...]:
        return self.effective_roles.effects

    @property
    def tested_causes(self) -> tuple[str, ...]:
        return self.effective_roles.tested_causes

    @property
    def cause_slices(self) -> Mapping[str, slice]:
        return self.feature_layout.cause_slices

    @property
    def predictor_slices(self) -> dict[str, slice]:
        return dict(self.cause_slices)

    @property
    def tested_cause_slices(self) -> dict[str, slice]:
        return {name: self.cause_slices[name] for name in self.tested_causes}

    @property
    def feature_lags(self) -> tuple[int, ...]:
        return self.feature_layout.feature_lags

    @property
    def final_lags(self) -> LagSpec:
        """Selected lags for the effective outputs and predictors, reconstructed from the mask."""
        return {
            effect: {
                name: tuple(sorted(set(np.asarray(self.feature_lags[block])[self.base_mask[i, block]].tolist())))
                for name, block in self.cause_slices.items()
            }
            for i, effect in enumerate(self.effects)
        }

    def inverse_transform_targets(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != len(self.effects):
            raise DataValidationError("Targets must use the prepared output order and shape.")
        result = np.asarray(self.target_inverse(array.copy()))
        if result.shape != array.shape:
            raise DataValidationError("Target-scale inverse must preserve shape.")
        return result

    def resolved_split(self) -> DataSplit:
        return self.split

    def indices_for(self, part: str, *, require_nonempty: bool = False) -> np.ndarray:
        return self.split.indices_for(part, require_nonempty=require_nonempty)

    def arrays_for(
        self, part: str, *, scaled: bool = True, require_nonempty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = (self.X_scaled, self.y_scaled) if scaled else (self.X_raw, self.y_raw)
        indices = self.indices_for(part, require_nonempty=require_nonempty)
        return X[indices], y[indices]

    def split_snapshot(self) -> dict:
        return self.split.to_dict(n_rows=self.X_raw.shape[0])
