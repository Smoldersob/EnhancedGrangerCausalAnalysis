"""Composable preparation facade built around explicit preparation stages."""

from __future__ import annotations

import copy
import warnings

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Mapping

import numpy as np
import pandas as pd

from ..core.exceptions import DataValidationError
from .lag import (
    LagEngine,
    resolve_lag_engine_config,
    resolve_lag_selector,
)
from .missing import BaseImputer
from .prepare_data import PreparedData, RemovedVariable, VariableRoles
from .sampling import NoSampling, SamplingStrategy
from .scaling import resolve_scaler, BlockScaler
from .stationarity import StationarityTransformer

_UNSET = object()

@dataclass(frozen=True)
class DataPreparationConfig:
    """Configuration shared by the staged preparation builder."""

    schema_policy: str = "error"
    backend_sample_fraction: float = 1.0
    backend_max_samples: Optional[int] = None
    random_state: Optional[int] = None
    x_scaler: Any = "standard"
    y_scaler: Any = "standard"
    lag_config: Any = None
    lag_selector: Any = None
    lag_selector_params: Mapping[str, Any] = field(
        default_factory=dict
    )
    lag_n_jobs: int = 1
    stationarity_transformer: Optional[StationarityTransformer] = None
    missing_value_imputer: Optional[BaseImputer] = None
    sampling_strategy: Optional[SamplingStrategy] = field(
        default_factory=NoSampling
    )

def _to_dataframe_list(data: pd.DataFrame | Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        return [data.copy()]
    if isinstance(data, Sequence) and all(isinstance(df, pd.DataFrame) for df in data):
        return [df.copy() for df in data]
    raise DataValidationError("data must be a DataFrame or a sequence of DataFrames")


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
        self._roles_assigned = False
        self._redundancy_removed = False

    @property
    def effective_roles(self) -> VariableRoles:
        return self.prepared_data_.effective_roles

    @property
    def requested_roles(self) -> VariableRoles:
        return self.prepared_data_.requested_roles
        
    def set_data(
        self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
    ) -> "DataPreparer":
        data_list = _to_dataframe_list(data)
        if not data_list:
            raise DataValidationError("The input data list is empty.")

        for index, dataframe in enumerate(data_list):
            if dataframe.columns.duplicated().any():
                raise DataValidationError(
                    f"DataFrame at index {index} contains "
                    "duplicated column names."
                )

        self.data_list = data_list
        self.prepared_data_ = PreparedData.empty()
        self._roles_assigned = False
        self._redundancy_removed = False
        return self

    def set_roles(
        self,
        *,
        causes: Optional[Sequence[str]] = None,
        tested_causes: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
    ) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError(
                "set_data() must be called before set_roles()."
            )
        if self._redundancy_removed:
            raise RuntimeError(
                "Roles cannot be reassigned after remove_redundant(). "
                "Start again with set_data()."
            )
        if self.prepared_data_.lag_engine is not None:
            raise RuntimeError(
                "Roles must be assigned before lag()."
            )

        columns = tuple(dict.fromkeys(
            column
            for dataframe in self.data_list
            for column in dataframe.columns
        ))
        available = set(columns)

        def normalize(values, default):
            if values is None:
                return tuple(default)
            if isinstance(values, str):
                values = (values,)
            result = tuple(values)
            return result if result else tuple(default)

        cause_names = normalize(causes, columns)
        effect_names = normalize(effects, columns)
        tested_names = normalize(tested_causes, cause_names)

        for role_name, names in (
            ("causes", cause_names),
            ("effects", effect_names),
            ("tested_causes", tested_names),
        ):
            if len(names) != len(set(names)):
                raise DataValidationError(
                    f"{role_name} contains duplicated names."
                )

            unknown = [name for name in names if name not in available]
            if unknown:
                raise DataValidationError(
                    f"Unknown variables in {role_name}: {unknown}"
                )

        cause_set = set(cause_names)
        invalid_tested = [
            name for name in tested_names if name not in cause_set
        ]
        if invalid_tested:
            raise DataValidationError(
                "tested_causes must be a subset of causes. "
                f"Variables outside causes: {invalid_tested}"
            )

        external_effects = [
            name for name in effect_names if name not in cause_set
        ]
        if external_effects:
            warnings.warn(
                "Some effects are not included in causes. "
                "Their lagged values will not be model inputs: "
                f"{external_effects}. This configuration is allowed.",
                UserWarning,
                stacklevel=2,
            )

        roles = VariableRoles(
            effects=effect_names,
            causes=cause_names,
            tested_causes=tested_names,
        )

        self.prepared_data_.requested_roles = roles
        self.prepared_data_.effective_roles = roles

        self._roles_assigned = True
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

    def imput(self, imputer: Any = _UNSET) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError("set_data() must be called before imput().")
        if self.prepared_data_.lag_engine is not None:
            raise RuntimeError("imput() must be called before lag().")

        effective = (
            self.config.missing_value_imputer
            if imputer is _UNSET
            else imputer
        )
        result = [df.copy() for df in self.data_list]
        if effective is not None:
            result = effective.fit_transform(result)

        self.data_list = [df.copy() for df in result]
        return self

    def remove_redundant(self) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError(
                "set_data() must be called before remove_redundant()."
            )
        if not self._roles_assigned:
            raise RuntimeError(
                "set_roles() must be called before remove_redundant()."
            )
        if self.prepared_data_.lag_engine is not None:
            raise RuntimeError(
                "remove_redundant() must be called before lag()."
            )
        if self._redundancy_removed:
            return self

        roles = self.prepared_data_.effective_roles
        required_columns = list(dict.fromkeys(
            (*roles.causes, *roles.effects)
        ))

        # Select relevant columns before redundancy diagnostics and lagging.
        data_list = [
            dataframe.loc[:, required_columns].copy()
            for dataframe in self.data_list
        ]

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

        causes_list = list(roles.causes)
        effects_list = list(roles.effects)
        tested_causes_list = list(roles.tested_causes)

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

        self.prepared_data_.removed_variables = {
            name: RemovedVariable(
                name=name,
                reason=str(info["reason"]),
                removed_roles=tuple(info["roles"]),
                retained_variable=info.get("retained_column"),
            )
            for name, info in removed_variable_info.items()
        }

        effective = VariableRoles(
            effects=tuple(effects_list),
            causes=tuple(causes_list),
            tested_causes=tuple(tested_causes_list),
        )

        self.prepared_data_.effective_roles = effective

        remaining_columns = list(dict.fromkeys(
            (*effective.causes, *effective.effects)
        ))
        self.data_list = [
            dataframe.loc[:, remaining_columns].copy()
            for dataframe in cleaned_data
        ]

        self._redundancy_removed = True
        return self
    
    def force_stationarity(
        self,
        transformer: Any = _UNSET,
    ) -> "DataPreparer":
        if self.data_list is None:
            raise RuntimeError(
                "set_data() must be called before force_stationarity()."
            )
        if self.prepared_data_.lag_engine is not None:
            raise RuntimeError(
                "force_stationarity() must be called before lag()."
            )

        effective = (
            self.config.stationarity_transformer
            if transformer is _UNSET
            else transformer
        )

        result = (
            [df.copy() for df in self.data_list]
            if effective is None
            else effective.fit_transform(self.data_list)
        )

        self.data_list = [df.copy() for df in result]
        self.prepared_data_.stationarity_transformer = effective
        return self

    def scale(
        self,
        *,
        x_scaler: Any = _UNSET,
        y_scaler: Any = _UNSET,
    ) -> "DataPreparer":
        prepared = self.prepared_data_
        if prepared.lag_engine is None:
            raise RuntimeError("lag() must be called before scale().")

        x_spec = self.config.x_scaler if x_scaler is _UNSET else x_scaler
        y_spec = self.config.y_scaler if y_scaler is _UNSET else y_scaler

        x_obj = copy.deepcopy(resolve_scaler(x_spec))
        y_obj = copy.deepcopy(resolve_scaler(y_spec))

        X_scaled = x_obj.fit_transform(prepared.X_raw)
        y_scaled = y_obj.fit_transform(prepared.y_raw)

        prepared.x_scaler = x_obj
        prepared.y_scaler = y_obj
        prepared.X_scaled = X_scaled
        prepared.y_scaled = y_scaled
        return self

    def block_scale(
        self,
        *,
        x_scaler: Any = _UNSET,
        y_scaler: Any = _UNSET,
    ) -> "DataPreparer":
        prepared = self.prepared_data_
        if prepared.lag_engine is None:
            raise RuntimeError(
                "lag() must be called before block_scale()."
            )

        x_spec = self.config.x_scaler if x_scaler is _UNSET else x_scaler
        y_spec = self.config.y_scaler if y_scaler is _UNSET else y_scaler

        x_obj = BlockScaler(
            blocks=prepared.cause_slices,
            scaler=x_spec,
        )
        y_obj = copy.deepcopy(resolve_scaler(y_spec))
        y_obj = copy.deepcopy(resolve_scaler(y_spec))

        X_scaled = x_obj.fit_transform(prepared.X_raw)
        y_scaled = y_obj.fit_transform(prepared.y_raw)

        prepared.x_scaler = x_obj
        prepared.y_scaler = y_obj
        prepared.X_scaled = X_scaled
        prepared.y_scaled = y_scaled
        return self

    def reduce_workload(
        self,
        *,
        backend_sample_fraction: Any = _UNSET,
        backend_max_samples: Any = _UNSET,
        random_state: Any = _UNSET,
        sampling_strategy: Any = _UNSET,
    ) -> "DataPreparer":
        prepared = self.prepared_data_
        if prepared.lag_engine is None:
            raise RuntimeError(
                "lag() must be called before reduce_workload()."
            )

        fraction = (
            self.config.backend_sample_fraction
            if backend_sample_fraction is _UNSET
            else backend_sample_fraction
        )
        maximum = (
            self.config.backend_max_samples
            if backend_max_samples is _UNSET
            else backend_max_samples
        )
        seed = (
            self.config.random_state
            if random_state is _UNSET
            else random_state
        )
        strategy = (
            self.config.sampling_strategy
            if sampling_strategy is _UNSET
            else sampling_strategy
        )
        if strategy is None:
            strategy = NoSampling()

        if not isinstance(strategy, SamplingStrategy):
            raise TypeError(
                "sampling_strategy must be a SamplingStrategy or None."
            )
        if (
            isinstance(fraction, (bool, np.bool_))
            or not isinstance(fraction, (int, float, np.number))
            or not 0 < fraction <= 1
        ):
            raise DataValidationError(
                "backend_sample_fraction must be in (0, 1]."
            )
        if maximum is not None and (
            isinstance(maximum, (bool, np.bool_))
            or not isinstance(maximum, (int, np.integer))
            or maximum <= 0
        ):
            raise DataValidationError(
                "backend_max_samples must be a positive integer or None."
            )
        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise DataValidationError(
                "random_state must be a non-negative integer or None."
            )

        n = prepared.X_raw.shape[0]
        fields = ["X_raw", "y_raw"]
        if prepared.x_scaler is not None:
            fields.append("X_scaled")
        if prepared.y_scaler is not None:
            fields.append("y_scaled")

        for name in fields:
            if getattr(prepared, name).shape[0] != n:
                raise DataValidationError(
                    f"{name} is not aligned with the prepared data."
                )

        source = prepared.sample_indices
        if source.shape != (n,):
            raise DataValidationError(
                "sample_indices is not aligned with the prepared data."
            )

        if n == 0:
            return self

        n_keep = max(1, int(np.floor(fraction * n)))
        if maximum is not None:
            n_keep = min(n_keep, int(maximum))

        indices = np.asarray(strategy.select(
            n,
            n_keep=n_keep,
            random_state=np.random.default_rng(seed),
        ))

        if (
            indices.ndim != 1
            or indices.dtype.kind not in {"i", "u"}
            or indices.size == 0
        ):
            raise DataValidationError(
                "Sampling must return a non-empty 1D integer index array."
            )
        if (
            np.any(indices >= n)
            or (indices.dtype.kind == "i" and np.any(indices < 0))
            or np.any(indices[1:] <= indices[:-1])
        ):
            raise DataValidationError(
                "Sampling indices must be valid, unique and increasing."
            )

        indices = indices.astype(np.intp, copy=False)
        for name in fields:
            setattr(prepared, name, getattr(prepared, name)[indices])

        prepared.sample_indices = source[indices].copy()
        return self

    def lag(
        self,
        *,
        lag_config: Any = _UNSET,
        lag_selector: Any = _UNSET,
        lag_selector_params: Any = _UNSET,
        n_jobs: Any = _UNSET,
    ) -> "DataPreparer":
        """Prepare lagged data and update effective predictor roles."""
        if self.data_list is None:
            raise RuntimeError(
                "set_data() must be called before lag()."
            )

        if not self._roles_assigned:
            raise RuntimeError(
                "set_roles() must be called before lag()."
            )

        prepared = self.prepared_data_

        if prepared.lag_engine is not None:
            raise RuntimeError(
                "lag() has already been applied. "
                "Start again with set_data() to rebuild preparation."
            )

        roles = prepared.effective_roles

        # These checks concern roles available before lag selection.
        # Lag selection may subsequently remove every tested cause.
        for role_name in ("effects", "causes", "tested_causes"):
            if not getattr(roles, role_name):
                raise DataValidationError(
                    f"No effective {role_name} are available before lagging."
                )

        if not set(roles.tested_causes).issubset(roles.causes):
            raise DataValidationError(
                "Effective tested_causes must be a subset of causes."
            )

        # Resolve engine configuration.
        raw_config = (
            self.config.lag_config
            if lag_config is _UNSET
            else lag_config
        )

        config = resolve_lag_engine_config(raw_config)

        if lag_selector is _UNSET:
            selector_spec = self.config.lag_selector
            selector_params = (
                self.config.lag_selector_params
                if lag_selector_params is _UNSET
                else lag_selector_params
            )
        else:
            # Replacing the selector also replaces its separate parameters
            # unless parameters are explicitly supplied in this call.
            selector_spec = lag_selector
            selector_params = (
                {}
                if lag_selector_params is _UNSET
                else lag_selector_params
            )

        if selector_params is None:
            selector_params = {}

        if not isinstance(selector_params, Mapping):
            raise TypeError(
                "lag_selector_params must be a mapping or None."
            )

        if not all(
            isinstance(name, str)
            for name in selector_params
        ):
            raise TypeError(
                "lag_selector_params keys must be strings."
            )

        selector = resolve_lag_selector(
            selector_spec,
            **dict(selector_params),
        )

        effective_n_jobs = (
            self.config.lag_n_jobs
            if n_jobs is _UNSET
            else n_jobs
        )

        engine = LagEngine(
            config=config,
            selector=selector,
            n_jobs=effective_n_jobs,
        )

        result = engine.prepare(
            self.data_list,
            outputs=roles.effects,
            predictors=roles.causes,
        )

        if result.X.shape[0] == 0 or result.y.shape[0] == 0:
            raise DataValidationError(
                "No observations remain after lag preparation."
            )

        if result.X.shape[1] == 0 or result.y.shape[1] == 0:
            raise DataValidationError(
                "No observations variables remain after lag preparation."
            )

        active_causes = tuple(result.active_predictors)
        removed_causes = tuple(result.removed_predictors)

        # Defensive checks at the pipeline/engine integration boundary.
        original_cause_set = set(roles.causes)
        active_set = set(active_causes)
        removed_set = set(removed_causes)

        if (
            active_set & removed_set
            or active_set | removed_set != original_cause_set
        ):
            raise DataValidationError(
                "LagEngine returned an inconsistent predictor partition."
            )

        if result.output_names != roles.effects:
            raise DataValidationError(
                "LagEngine changed the requested output order."
            )

        active_causes = result.active_predictors
        removed_causes = result.removed_predictors

        boundaries = result.block_boundaries

        if len(boundaries) != len(active_causes) + 1:
            raise DataValidationError(
                "LagEngine returned invalid predictor block boundaries."
            )

        cause_slices = {
            name: slice(int(start), int(stop))
            for name, start, stop in zip(
                active_causes,
                boundaries[:-1],
                boundaries[1:],
            )
        }

        effective_roles = VariableRoles(
            effects=roles.effects,
            causes=active_causes,
            tested_causes=tuple(
                name
                for name in roles.tested_causes
                if name in active_set
            ),
        )

        # Preserve constant/duplicate records created earlier.
        removed_variables = dict(prepared.removed_variables)

        for name in removed_causes:
            if name in removed_variables:
                raise DataValidationError(
                    f"Variable {name!r} already has removal metadata, "
                    "but was still supplied as an effective predictor."
                )

            removed_roles = ["cause"]

            if name in roles.tested_causes:
                removed_roles.append("tested_cause")

            removed_variables[name] = RemovedVariable(
                name=name,
                reason="no_selected_lags",
                removed_roles=tuple(removed_roles),
                retained_variable=None,
            )

        # Commit only after engine execution and consistency checks succeed.
        prepared.lag_engine = engine
        prepared.lag_result = result

        prepared.effective_roles = effective_roles
        prepared.removed_variables = removed_variables

        prepared.X_raw = result.X.copy()
        prepared.y_raw = result.y.copy()

        prepared.base_mask = np.asarray(
            result.mask,
            dtype=np.float64,
        ).copy()

        prepared.feature_lags = result.feature_lags.copy()
        prepared.cause_slices = cause_slices

        prepared.sample_indices = np.arange(
            result.X.shape[0],
            dtype=int,
        )


        return self

    def fit(
    self,
        data: pd.DataFrame | Sequence[pd.DataFrame],
        roles: Optional[VariableRoles] = None,
    ) -> "DataPreparer":
        self.set_data(data)

        if roles is None:
            self.set_roles()
        else:
            self.set_roles(
                causes=roles.causes,
                tested_causes=roles.tested_causes,
                effects=roles.effects,
            )

        (
            self
            .align()
            .imput()
            .remove_redundant()
            .force_stationarity()
            .lag()
            .scale()
            .reduce_workload()
        )
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
