from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ValueType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    COUNT = "count"


class ProcessRole(str, Enum):
    PROCESS_VARIABLE = "process_variable"
    TERMINAL_OUTPUT = "terminal_output"
    REDUNDANT_MEASUREMENT = "redundant_measurement"
    VIRTUAL_MEASUREMENT = "virtual_measurement"
    SETPOINT = "setpoint"
    MANUAL_SETPOINT = "manual_setpoint"
    CONTROL_SIGNAL = "control_signal"
    MANIPULATED_VARIABLE = "manipulated_variable"
    CONTROL_ERROR = "control_error"
    CONTROLLED_VARIABLE = "controlled_variable"
    FAULT = "fault"
    DIAGNOSTIC_SIGNAL = "diagnostic_signal"
    RESIDUAL = "residual"
    DISTURBANCE = "disturbance"


@dataclass(frozen=True)
class VariableDescription:
    """Semantic description of a process variable."""

    name: str
    value_type: ValueType
    process_roles: tuple[ProcessRole, ...] = ()
    domain_constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Variable name must be a non-empty string.")

        value_type = self.value_type
        if isinstance(value_type, str):
            value_type = ValueType(value_type)
        if not isinstance(value_type, ValueType):
            raise TypeError("value_type must be ValueType or a supported name.")

        roles: list[ProcessRole] = []
        for role in self.process_roles:
            normalized = ProcessRole(role) if isinstance(role, str) else role
            if not isinstance(normalized, ProcessRole):
                raise TypeError("process_roles must contain ProcessRole values.")
            if normalized not in roles:
                roles.append(normalized)

        if not isinstance(self.domain_constraints, Mapping):
            raise TypeError("domain_constraints must be a mapping.")

        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "process_roles", tuple(roles))
        object.__setattr__(
            self,
            "domain_constraints",
            MappingProxyType(dict(self.domain_constraints)),
        )
