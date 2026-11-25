"""Fitted preprocessing state and transform-only execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.data.feature_layout import FeatureLayout
from ..core.data.prepared import PreparedData
from ..core.data.split import DataSplit
from ..core.exceptions import DataValidationError
from ..preprocessing.lag.artifacts import LaggedData
from .configuration.plan import FittingPlan
from .splitting.scope import FitScope


@dataclass(frozen=True)
class ApplyOptions:
    """Per-call output assignment; never changes the fitted plan."""

    split: DataSplit | None = None


@dataclass(frozen=True)
class PreparationReport:
    """Stable provenance for decisions made by one fit."""

    scope: FitScope
    schema: tuple[str, ...]
    final_lags: Mapping[str, Mapping[str, tuple[int, ...]]]
    removed_variables: Mapping[str, Any]


@dataclass(frozen=True)
class FittedPreprocessor:
    """Reusable fitted components; transform never estimates decisions."""

    plan: FittingPlan
    scope: FitScope
    input_columns: tuple[str, ...]
    effective_roles: Any
    removed_variables: Mapping[str, Any]
    feature_layout: FeatureLayout
    base_mask: np.ndarray
    _lag_fitted: Any
    _imputer: Any
    _stationarity: Any
    _lag_engine: Any
    _x_scaler: Any
    _y_scaler: Any
    report: PreparationReport

    @classmethod
    def from_executor(
        cls,
        executor: Any,
        plan: FittingPlan,
        scope: FitScope,
        training_data: PreparedData,
    ) -> "FittedPreprocessor":
        state = executor.prepared_data_
        if executor.data_list is None or state.lag_fitted is None:
            raise RuntimeError("The preparation executor did not finish fitting.")
        if not isinstance(training_data, PreparedData):
            raise TypeError("training_data must be PreparedData.")
        final_lags = state.lag_fitted.final_lags
        final_lags_snapshot = {
            output: MappingProxyType({
                predictor: tuple(lags)
                for predictor, lags in predictors.items()
            })
            for output, predictors in final_lags.items()
        }
        removed = MappingProxyType(dict(state.removed_variables))
        report = PreparationReport(
            scope=scope,
            schema=plan.spec.schema.columns,
            final_lags=MappingProxyType(final_lags_snapshot),
            removed_variables=removed,
        )
        return cls(
            plan=plan,
            scope=scope,
            input_columns=tuple(executor.data_list[0].columns),
            effective_roles=state.effective_roles,
            removed_variables=removed,
            feature_layout=training_data.feature_layout,
            base_mask=training_data.base_mask,
            _lag_fitted=deepcopy(state.lag_fitted),
            _imputer=deepcopy(executor._fitted_imputer),
            _stationarity=deepcopy(state.stationarity_enforcer),
            _lag_engine=deepcopy(state.lag_engine),
            _x_scaler=deepcopy(state.x_scaler),
            _y_scaler=deepcopy(state.y_scaler),
            report=report,
        )

    def transform(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
        *,
        options: ApplyOptions | None = None,
    ) -> PreparedData:
        if options is not None and not isinstance(options, ApplyOptions):
            raise TypeError("options must be ApplyOptions or None.")
        frames = self._normalize_frames(data)
        expected = self.plan.spec.schema.columns
        missing = [name for name in expected if any(name not in frame.columns for frame in frames)]
        if missing and (
            self.plan.spec.config.schema_policy != "union"
            or self._imputer is None
        ):
            raise DataValidationError(
                "Transform input is missing required columns "
                f"{list(dict.fromkeys(missing))}; union schema requires a fitted imputer."
            )

        aligned = [
            frame.reindex(columns=expected).copy()
            for frame in frames
        ]
        selected = set(self.plan.spec.requested_roles.causes) | set(
            self.plan.spec.requested_roles.effects
        )
        for frame_index, frame in enumerate(aligned):
            for name in selected:
                values = frame[name]
                if (
                    not pd.api.types.is_numeric_dtype(values.dtype)
                    or pd.api.types.is_bool_dtype(values.dtype)
                ):
                    raise DataValidationError(
                        f"Variable {name!r} no longer matches the fitted "
                        f"numeric-continuous strategy in DataFrame {frame_index}."
                    )
        imputer = self._copy_component(self._imputer, "imputer")
        if imputer is not None:
            aligned = imputer.transform(aligned)

        aligned = [
            frame.drop(
                columns=[
                    name for name, record in self.removed_variables.items()
                    if record.reason in {"constant", "duplicate"}
                ],
                errors="ignore",
            )
            for frame in aligned
        ]
        aligned = [frame.loc[:, self.input_columns].copy() for frame in aligned]

        stationarity = self._copy_component(self._stationarity, "stationarity")
        if stationarity is not None:
            aligned = stationarity.transform(aligned)

        lag_engine = self._copy_component(self._lag_engine, "lag engine")
        lagged = lag_engine.transform(aligned)
        x_scaler = self._copy_component(self._x_scaler, "X scaler")
        y_scaler = self._copy_component(self._y_scaler, "Y scaler")
        X_scaled = x_scaler.transform(lagged.X)
        y_scaled = y_scaler.transform(lagged.y)

        segments = tuple(
            type(segment)(
                dataset_index=segment.dataset_index,
                original_row_count=len(frames[segment.dataset_index]),
                dropped_leading_rows=segment.dropped_leading_rows,
                prepared_row_slice=segment.prepared_row_slice,
                retained_index=segment.retained_index,
            )
            for segment in lagged.segments
        )
        split = (
            options.split
            if options is not None and options.split is not None
            else DataSplit.all_training(lagged.X.shape[0])
        )
        return PreparedData(
            X_raw=lagged.X,
            y_raw=lagged.y,
            X_scaled=X_scaled,
            y_scaled=y_scaled,
            feature_layout=self.feature_layout,
            requested_roles=self.plan.spec.requested_roles,
            effective_roles=self.effective_roles,
            base_mask=self.base_mask,
            split=split,
            sample_indices=np.arange(lagged.X.shape[0], dtype=np.intp),
            target_inverse=self._copy_component(
                self._y_scaler, "Y scaler"
            ).inverse_transform,
            segments=segments,
            variable_descriptions=self.plan.spec.variable_descriptions,
            removed_variables=self.removed_variables,
        )

    def _normalize_frames(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            frames = [data.copy()]
        elif isinstance(data, Sequence) and all(
            isinstance(frame, pd.DataFrame) for frame in data
        ):
            frames = [frame.copy() for frame in data]
        else:
            raise DataValidationError(
                "Transform data must be a DataFrame or a sequence of DataFrames."
            )
        if not frames:
            raise DataValidationError("Transform input must not be empty.")
        for index, frame in enumerate(frames):
            if frame.columns.duplicated().any():
                raise DataValidationError(
                    f"Transform DataFrame at index {index} contains duplicate columns."
                )
        return frames

    @staticmethod
    def _copy_component(component: Any, name: str) -> Any:
        if component is None:
            return None
        try:
            return deepcopy(component)
        except Exception as exc:
            raise TypeError(
                f"Fitted {name} must support deepcopy to keep transform calls isolated."
            ) from exc

    @property
    def lag_engine(self) -> Any:
        """Return an isolated copy for inspection; the fitted prototype stays private."""
        return self._copy_component(self._lag_engine, "lag engine")

    @property
    def stationarity_enforcer(self) -> Any:
        """Return an isolated copy for inspection."""
        return self._copy_component(self._stationarity, "stationarity")


__all__ = ["ApplyOptions", "FittedPreprocessor", "PreparationReport"]
