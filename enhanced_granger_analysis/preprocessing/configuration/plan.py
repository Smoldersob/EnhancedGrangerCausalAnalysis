"""Value-independent contracts for a prepared-data workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

import pandas as pd

from ...core.data.analysis_roles import VariableRoles
from ...core.data.variables import ValueType, VariableDescription
from ...core.exceptions import DataValidationError
from .preparation import DataPreparationConfig


_VALUE_STRATEGIES = {
    ValueType.CONTINUOUS: "numeric-continuous",
}


@dataclass(frozen=True)
class SourceSchema:
    """Ordered input names only; schema planning never inspects values."""

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        if not columns or any(not isinstance(name, str) or not name for name in columns):
            raise DataValidationError("Source schema must contain non-empty column names.")
        if len(columns) != len(set(columns)):
            raise DataValidationError("Source schema contains duplicate column names.")
        object.__setattr__(self, "columns", columns)

    @classmethod
    def from_frames(
        cls,
        frames: pd.DataFrame | Sequence[pd.DataFrame],
        *,
        policy: str,
    ) -> "SourceSchema":
        frame_list = [frames] if isinstance(frames, pd.DataFrame) else list(frames)
        if not frame_list or not all(isinstance(frame, pd.DataFrame) for frame in frame_list):
            raise DataValidationError("Planning requires one or more DataFrames.")
        if any(frame.columns.duplicated().any() for frame in frame_list):
            raise DataValidationError("Source data contains duplicate column names.")
        first = list(frame_list[0].columns)
        if policy == "error":
            for index, frame in enumerate(frame_list[1:], start=1):
                if list(frame.columns) != first:
                    raise DataValidationError(
                        f"DataFrame at index {index} has columns "
                        f"{list(frame.columns)}, expected {first}."
                    )
            columns = first
        elif policy == "union":
            columns = list(dict.fromkeys(
                name for frame in frame_list for name in frame.columns
            ))
        else:
            raise DataValidationError("schema_policy must be 'error' or 'union'.")
        return cls(tuple(columns))


@dataclass(frozen=True)
class ResolvedSpec:
    """Bound declarations; contains no observed values or fitted objects."""

    config: DataPreparationConfig
    schema: SourceSchema
    requested_roles: VariableRoles
    variable_descriptions: Mapping[str, VariableDescription]
    value_strategies: Mapping[str, str]


@dataclass(frozen=True)
class FittingPlan:
    """Ordered preprocessing phases and value-independent decisions."""

    spec: ResolvedSpec
    steps: tuple[str, ...] = (
        "schema",
        "imputation",
        "diagnostics",
        "stationarity",
        "lag-selection",
        "lag-construction",
        "scaling",
    )
    deferred_decisions: tuple[str, ...] = (
        "diagnostic-facts",
        "representation-policy",
        "lag-selection",
    )


def _snapshot_config(config: DataPreparationConfig) -> DataPreparationConfig:
    """Detach a plan from caller-owned mutable configuration values."""
    descriptions = {
        name: VariableDescription(
            description.name,
            description.value_type,
            description.process_roles,
            dict(description.domain_constraints),
        )
        for name, description in config.variable_descriptions.items()
    }
    try:
        return replace(
            config,
            variable_descriptions=descriptions,
            lag_selector_params=deepcopy(dict(config.lag_selector_params)),
            missing_value_imputer=deepcopy(config.missing_value_imputer),
            splitter=deepcopy(config.splitter),
            sampling_strategy=deepcopy(config.sampling_strategy),
            lag_config=deepcopy(config.lag_config),
            lag_selector=deepcopy(config.lag_selector),
            stationarity=deepcopy(config.stationarity),
            x_scaler=deepcopy(config.x_scaler),
            y_scaler=deepcopy(config.y_scaler),
        )
    except Exception as exc:
        raise TypeError(
            "Preparation configuration components must support deepcopy "
            "to create an isolated fitting plan."
        ) from exc


def plan_preparation(
    config: DataPreparationConfig,
    schema: SourceSchema | pd.DataFrame | Sequence[pd.DataFrame],
    roles: VariableRoles | None = None,
) -> FittingPlan:
    """Bind roles and value strategies to an ordered source schema."""
    if not isinstance(config, DataPreparationConfig):
        raise TypeError("config must be a DataPreparationConfig.")
    if not isinstance(schema, SourceSchema):
        schema = SourceSchema.from_frames(schema, policy=config.schema_policy)

    columns = schema.columns
    available = set(columns)
    if roles is None:
        requested = VariableRoles(
            effects=columns,
            causes=columns,
            tested_causes=columns,
        )
    elif isinstance(roles, VariableRoles):
        requested = roles
    else:
        raise TypeError("roles must be VariableRoles or None.")

    for role_name, names in (
        ("effects", requested.effects),
        ("causes", requested.causes),
        ("tested_causes", requested.tested_causes),
    ):
        if role_name != "tested_causes" and not names:
            raise DataValidationError(f"{role_name} must not be empty in a preparation plan.")
        if len(names) != len(set(names)):
            raise DataValidationError(f"{role_name} contains duplicate names.")
        unknown = [name for name in names if name not in available]
        if unknown:
            raise DataValidationError(f"Unknown variables in {role_name}: {unknown}.")
    if not set(requested.tested_causes).issubset(requested.causes):
        raise DataValidationError("tested_causes must be a subset of causes.")

    descriptions = dict(config.variable_descriptions)
    for name, description in descriptions.items():
        if not isinstance(description, VariableDescription) or description.name != name:
            raise DataValidationError(
                "variable_descriptions must map names to matching VariableDescription objects."
            )
        if name not in available:
            raise DataValidationError(f"Description refers to unknown variable {name!r}.")

    strategies: dict[str, str] = {}
    for name in columns:
        description = descriptions.get(name)
        value_type = (
            description.value_type if description is not None
            else config.default_value_type
        )
        strategy = _VALUE_STRATEGIES.get(value_type)
        if name in set((*requested.causes, *requested.effects)) and strategy is None:
            raise DataValidationError(
                f"No preprocessing strategy is implemented for value type "
                f"{value_type.value!r} of variable {name!r}."
            )
        if strategy is not None:
            strategies[name] = strategy
        if description is None:
            descriptions[name] = VariableDescription(name, value_type)

    frozen_config = _snapshot_config(config)
    resolved = ResolvedSpec(
        config=frozen_config,
        schema=schema,
        requested_roles=requested,
        variable_descriptions=MappingProxyType(descriptions),
        value_strategies=MappingProxyType(strategies),
    )
    return FittingPlan(resolved)


__all__ = [
    "FittingPlan",
    "ResolvedSpec",
    "SourceSchema",
    "plan_preparation",
]
