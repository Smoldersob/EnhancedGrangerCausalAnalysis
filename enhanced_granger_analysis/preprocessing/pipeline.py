"""Composable preparation facade built around explicit preparation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import DataValidationError
from .lag import BaseLagSelector, LagConfiguration, LagEngine
from .missing import BaseImputer
from .prepare_data import PreparedData, RemovedVariable, VariableRoles
from .sampling import NoSampling, RandomSampling, SamplingStrategy
from .scaling.scalers import resolve_scaler
from .stationarity import StationarityTransformer


@dataclass(frozen=True)
class DataPreparationConfig:
    """Configuration shared by the staged preparation builder."""

    schema_policy: str = "error"
    backend_sample_fraction: float = 1.0
    backend_max_samples: Optional[int] = None
    random_state: Optional[int] = None
    x_scaler: Any = "standard"
    y_scaler: Any = "standard"
    lag_config: Optional[LagConfiguration] = None
    lag_selector: Optional[BaseLagSelector] = None
    stationarity_transformer: Optional[StationarityTransformer] = None
    missing_value_imputer: Optional[BaseImputer] = None
    sampling_strategy: Optional[SamplingStrategy] = None


def _to_dataframe_list(data: pd.DataFrame | Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        return [data.copy()]
    if isinstance(data, Sequence) and all(isinstance(df, pd.DataFrame) for df in data):
        return [df.copy() for df in data]
    raise DataValidationError("data must be a DataFrame or a sequence of DataFrames")


def _create_scaler(name: Optional[Any]) -> Any:
    return resolve_scaler(name)


class DataPreparer:
    """Stage-by-stage preparation builder for :class:`PreparedData`."""

    def __init__(
        self,
        config: Optional[DataPreparationConfig] = None,
        **config_overrides: Any,
    ) -> None:
        if config is not None and config_overrides:
            raise ValueError("Use config or config_overrides, not both.")
        self.config = config or DataPreparationConfig(**config_overrides)
        self.data_list: Optional[list[pd.DataFrame]] = None
        self.prepared_data_: Optional[PreparedData] = PreparedData.empty()

    @property
    def effective_roles(self) -> VariableRoles:
        return self.prepared_data_.effective_roles

    @property
    def requested_roles(self) -> VariableRoles:
        return self.prepared_data_.requested_roles

    @property
    def effects(self) -> list[str]:
        return self.prepared_data_.effects
        
    def set_data(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
    ) -> "DataPreparer":
        self.data_list = _to_dataframe_list(data)
        self.prepared_data_ = PreparedData.empty()
        return self

    def set_roles(
        self,
        *,
        causes: Optional[Sequence[str]] = None,
        tested_causes: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
    ) -> "DataPreparer":
        request_causes = tuple(causes) if causes is not None else ()
        request_tested = tuple(tested_causes) if tested_causes is not None else ()
        request_effects = tuple(effects) if effects is not None else ()
        self.prepared_data_.requested_roles = VariableRoles(
            effects=request_effects,
            causes=request_causes,
            tested_causes=request_tested,
        )
        self.prepared_data_.effective_roles = VariableRoles(
            effects=request_effects,
            causes=request_causes,
            tested_causes=request_tested,
        )
        return self

    def align(
        self,
        *,
        schema_policy: Optional[str] = None,
    ) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError("DataPreparer.set_data() must be called before align().")
        schema_policy = self.config.schema_policy if schema_policy is None else schema_policy
        if schema_policy not in {"error", "union"}:
            raise DataValidationError("schema_policy must be 'error' or 'union'.")
        aligned_data = [df.copy() for df in self.data_list]
        if schema_policy == "union":
            columns = list(dict.fromkeys(column for df in aligned_data for column in df.columns))
            aligned_data = [df.reindex(columns=columns).copy() for df in aligned_data]
        else:
            reference_columns = list(aligned_data[0].columns)
            for index, dataframe in enumerate(aligned_data[1:], start=1):
                if list(dataframe.columns) != reference_columns:
                    raise DataValidationError(
                        f"DataFrame at index {index} has columns {list(dataframe.columns)}, expected {reference_columns}"
                    )
        self.data_list = aligned_data
        return self

    def imput(self, imputer: Optional[BaseImputer] = None) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError("DataPreparer.set_data() must be called before imput().")
        imputer = self.config.missing_value_imputer if imputer is None else imputer
        imputed_data = [df.copy() for df in self.data_list]
        if imputer is not None:
            imputed_data = imputer.fit_transform(imputed_data)
        self.data_list = [df.copy() for df in imputed_data]
        return self

    def remove_redundant(
        self,
        *,
        causes: Optional[Sequence[str]] = None,
        tested_causes: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
    ) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError("DataPreparer.set_data() must be called before remove_redundant().")

        if causes is not None:
            self.prepared_data_.effective_roles = VariableRoles(
                effects= self.prepared_data_.effective_roles.effects,
                causes=tuple(causes),
                tested_causes=self.prepared_data_.effective_roles.tested_causes,
            )
        if tested_causes is not None:
            self.prepared_data_.effective_roles = VariableRoles(
                effects=self.prepared_data_.effective_roles.effects,
                causes=self.prepared_data_.effective_roles.causes,
                tested_causes=tuple(tested_causes),
            )
        if effects is not None:
            self.prepared_data_.effective_roles  = VariableRoles(
                effects=tuple(effects),
                causes=self.prepared_data_.effective_roles.causes,
                tested_causes=self.prepared_data_.effective_roles.tested_causes,
            )

        data_list = [df.copy() for df in self.data_list]
        all_columns = list(dict.fromkeys(column for dataframe in data_list for column in dataframe.columns))
        aligned_data = [df.astype(float).copy() for df in data_list]

        combined = pd.concat(aligned_data, axis=0, ignore_index=True)
        constant_columns = [
            column for column in all_columns if combined[column].nunique(dropna=False) <= 1
        ]
        data_without_constants = [df.drop(columns=constant_columns) for df in aligned_data]
        combined_without_constants = pd.concat(data_without_constants, axis=0, ignore_index=True)

        duplicate_columns_map: dict[str, str] = {}
        retained_columns: list[str] = []
        for column in combined_without_constants.columns:
            retained_match = None
            for retained_column in retained_columns:
                if combined_without_constants[column].equals(combined_without_constants[retained_column]):
                    retained_match = retained_column
                    break
            if retained_match is None:
                retained_columns.append(column)
            else:
                duplicate_columns_map[column] = retained_match

        removed_columns = set(constant_columns)
        removed_columns.update(duplicate_columns_map)

        cleaned_data = [df.drop(columns=list(removed_columns), errors="ignore") for df in data_without_constants]

        requested_causes = tuple(self.prepared_data_.requested_roles.causes) or tuple(all_columns)
        requested_effects = tuple(self.prepared_data_.requested_roles.effects) or tuple(all_columns)
        requested_tested = tuple(self.prepared_data_.requested_roles.tested_causes) or requested_causes
        causes_list = list(requested_causes)
        effects_list = list(requested_effects)
        tested_causes_list = list(requested_tested)

        unknown_tested_role = [column for column in tested_causes_list if column not in causes_list]
        if unknown_tested_role:
            raise DataValidationError(
                "tested_causes must be a subset of causes. Not present in causes: "
                f"{unknown_tested_role}"
            )

        unknown_causes = [column for column in causes_list if column not in all_columns]
        unknown_effects = [column for column in effects_list if column not in all_columns]
        unknown_tested_causes = [column for column in tested_causes_list if column not in all_columns]
        if unknown_causes or unknown_effects or unknown_tested_causes:
            raise DataValidationError(
                "Unknown variables detected. "
                f"causes={unknown_causes}, effects={unknown_effects}, tested_causes={unknown_tested_causes}"
            )

        removed_variable_info: dict[str, dict[str, Any]] = {}
        for column in constant_columns:
            roles: list[str] = []
            if column in effects_list:
                roles.append("effect")
            if column in causes_list:
                roles.append("cause")
            if column in tested_causes_list:
                roles.append("tested_cause")
            if roles:
                removed_variable_info[column] = {"reason": "constant", "roles": roles}

        for duplicate_column, retained_column in duplicate_columns_map.items():
            roles: list[str] = []
            if duplicate_column in effects_list:
                roles.append("effect")
            if duplicate_column in causes_list:
                roles.append("cause")
            if duplicate_column in tested_causes_list:
                roles.append("tested_cause")
            if roles:
                removed_variable_info[duplicate_column] = {
                    "reason": "duplicate",
                    "retained_column": retained_column,
                    "roles": roles,
                }

        causes_list = [column for column in causes_list if column not in removed_columns]
        tested_causes_list = [column for column in tested_causes_list if column not in removed_columns]
        effects_list = [column for column in effects_list if column not in removed_columns]

        self.prepared_data_.removed_variable_info = removed_variable_info
        self.prepared_data_.removed_variables = {
            name: RemovedVariable(
                name=name,
                reason=str(info["reason"]),
                removed_roles=tuple(info["roles"]),
                retained_variable=info.get("retained_column"),
            )
            for name, info in removed_variable_info.items()
        }
        self.prepared_data_.effective_roles = VariableRoles(
            effects=tuple(effects_list),
            causes=tuple(causes_list),
            tested_causes=tuple(tested_causes_list),
        )
        self.data_list = [df.copy() for df in cleaned_data]

        return self

    def force_stationarity(
        self,
        transformer: Optional[StationarityTransformer] = None,
    ) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError("DataPreparer.set_data() must be called before force_stationarity().")
        transformer = self.config.stationarity_transformer if transformer is None else transformer
        if transformer is None:
            stationary_data = [df.copy() for df in self.data_list]
        else:
            stationary_data = transformer.fit_transform(self.data_list)
        self.prepared_data_.stationarity_transformer = transformer
        self.data_list = [df.copy() for df in stationary_data]
        return self

    def scale(
        self,
        *,
        x_scaler: Optional[Any] = None,
        y_scaler: Optional[Any] = None,
    ) -> "DataPreparer":
        self._x_scaler = self.config.x_scaler if x_scaler is None else x_scaler
        self._y_scaler = self.config.y_scaler if y_scaler is None else y_scaler
        if self.prepared_data_ is None or self.prepared_data_.lag_engine is None:
            return self
        x_scaler_obj = _create_scaler(self._x_scaler)
        y_scaler_obj = _create_scaler(self._y_scaler)
        self.prepared_data_.x_scaler = x_scaler_obj
        self.prepared_data_.y_scaler = y_scaler_obj
        self.prepared_data_.X_scaled = x_scaler_obj.fit_transform(
            self.prepared_data_.X_raw
        )
        self.prepared_data_.y_scaled = y_scaler_obj.fit_transform(
            self.prepared_data_.y_raw
        )
        return self

    def reduce_workload(
        self,
        *,
        backend_sample_fraction: Optional[float] = None,
        backend_max_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        sampling_strategy: Optional[SamplingStrategy] = NoSampling(),
    ) -> "DataPreparer":
        backend_sample_fraction = self.config.backend_sample_fraction if backend_sample_fraction is None else backend_sample_fraction
        backend_max_samples = self.config.backend_max_samples if backend_max_samples is None else backend_max_samples
        random_state = self.config.random_state if random_state is None else random_state
        sampling_strategy = self.config.sampling_strategy if sampling_strategy is None else sampling_strategy

        if not 0 < backend_sample_fraction <= 1:
            raise DataValidationError("backend_sample_fraction must be in the interval (0, 1].")
        if backend_max_samples is not None and backend_max_samples <= 0:
            raise DataValidationError("backend_max_samples must be positive.")
        if random_state is not None and not isinstance(random_state, (int, np.integer)):
            raise DataValidationError("random_state must be an integer or None.")
        if self.prepared_data_ is None or self.prepared_data_.lag_engine is None:
            return self
        if not isinstance(self.prepared_data_.X_raw, np.ndarray) or not isinstance(self.prepared_data_.y_raw, np.ndarray):
            raise TypeError("X_raw and y_raw must be numpy arrays")
        if self.prepared_data_.X_raw.size == 0 or self.prepared_data_.y_raw.size == 0:
            self.prepared_data_.sample_indices = np.empty(0, dtype=int)
            return self
        
        n = self.prepared_data_.X_raw.shape[0]
        n_keep = max(1, int(np.floor(backend_sample_fraction * n)))
        if backend_max_samples is not None:
            n_keep = min(n_keep, backend_max_samples)
        strategy = sampling_strategy or self.config.sampling_strategy or RandomSampling()
        idx = strategy.select(
            n,
            n_keep=n_keep,
            random_state=np.random.default_rng(random_state),
        )
        self.prepared_data_.sample_indices = idx

        self.prepared_data_.X_raw = self.prepared_data_.X_raw[idx]
        self.prepared_data_.y_raw = self.prepared_data_.y_raw[idx]

        return self

    def lag(
        self,
        *,
        lag_config: Optional[LagConfiguration] = None,
        lag_selector: Optional[BaseLagSelector] = None,
    ) -> 'DataPreparer':
        if self.data_list is None:
            raise RuntimeError("DataPreparer.set_data() must be called before lag().")

        data_list = [df.copy() for df in self.data_list]
        if not self.prepared_data_.effective_roles.causes and not self.prepared_data_.effective_roles.effects and not self.prepared_data_.effective_roles.tested_causes:
            all_columns = data_list[0].columns.tolist()
            
            self.prepared_data_.effective_roles = VariableRoles(
                effects=tuple(all_columns),
                causes=tuple(all_columns),
                tested_causes=tuple(all_columns),
            )

        if not self.prepared_data_.effective_roles.effects:
            causes_list = list(self.prepared_data_.effective_roles.causes)
            effects_list = list(self.prepared_data_.effective_roles.effects) or list(self.data_list[0].columns)
            tested_causes_list = list(self.prepared_data_.effective_roles.tested_causes) or causes_list
            self.prepared_data_.effective_roles = VariableRoles(
                effects=tuple(effects_list),
                causes=tuple(causes_list),
                tested_causes=tuple(tested_causes_list),
            )

        engine = LagEngine(
            config=lag_config or self.config.lag_config or LagConfiguration(),
            selector=lag_selector or self.config.lag_selector,
        )
        
        X_raw, y_raw, col_offsets = engine.prepare(data_list, effects=list(self.prepared_data_.effective_roles.effects))

        all_columns = data_list[0].columns.tolist()
        base_mask = None
        if engine.mask_ is not None:
            mask_arr = np.asarray(engine.mask_, dtype=np.float64)
            if mask_arr.shape[0] == len(self.prepared_data_.effective_roles.effects):
                base_mask = mask_arr
            else:
                effect_indices = [
                    all_columns.index(effect_name)
                    for effect_name in self.prepared_data_.effective_roles.effects
                ]
                base_mask = mask_arr[effect_indices, :]

        predictor_slices = {
            column: slice(int(col_offsets[index]), int(col_offsets[index + 1]))
            for index, column in enumerate(all_columns)
        }
        self.prepared_data_.base_mask = base_mask
        self.prepared_data_.lag_engine = engine
        self.prepared_data_.col_offsets = col_offsets
        self.prepared_data_.X_raw = X_raw
        self.prepared_data_.y_raw = y_raw
        self.prepared_data_.causes = list(self.prepared_data_.effective_roles.causes)
        self.prepared_data_.tested_causes = list(
            self.prepared_data_.effective_roles.tested_causes
        )
        self.prepared_data_.effects = list(self.prepared_data_.effective_roles.effects)
        self.prepared_data_.predictor_slices = {
            cause: predictor_slices[cause]
            for cause in self.prepared_data_.effective_roles.causes
        }
        self.prepared_data_.cause_slices = {
            cause: predictor_slices[cause]
            for cause in self.prepared_data_.effective_roles.causes
        }
        self.prepared_data_.tested_cause_slices = {
            cause: predictor_slices[cause]
            for cause in self.prepared_data_.effective_roles.tested_causes
        }
        return self

    def fit(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
        roles: Optional[VariableRoles] = None,
    ) -> "DataPreparer":
        self.set_data(data)
        if roles is not None:
            self.set_roles(
                causes=roles.causes,
                tested_causes=roles.tested_causes,
                effects=roles.effects,
            )
        self.align().imput().remove_redundant().force_stationarity().lag().reduce_workload().scale()
        return self
    
    def get_feature_map(self) -> dict[str, slice]:
        if self.prepared_data_ is None:
            raise RuntimeError("DataPreparer is not fitted. Call fit() first.")
        return dict(self.prepared_data_.cause_slices)

    def get_report(self) -> dict[str, Any]:
        if self.prepared_data_ is None:
            raise RuntimeError("DataPreparer is not fitted. Call fit() first.")
        return {
            "n_samples": int(self.prepared_data_.X_raw.shape[0]),
            "n_features": int(self.prepared_data_.X_raw.shape[1]),
            "effects": self.prepared_data_.effective_roles.effects,
            "causes": self.prepared_data_.effective_roles.causes,
            "tested_causes": self.prepared_data_.effective_roles.tested_causes,
        }


__all__ = ["DataPreparationConfig", "DataPreparer"]
