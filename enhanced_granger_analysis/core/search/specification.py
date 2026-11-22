from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .objective import Direction, ObjectiveScale


@dataclass(frozen=True)
class SearchComponentSpec:
    selector: Any
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise TypeError("Component params must be a mapping.")
        if not all(isinstance(key, str) for key in self.params):
            raise TypeError("Component parameter names must be strings.")
        object.__setattr__(
            self,
            "params",
            MappingProxyType(dict(self.params)),
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    component: SearchComponentSpec = field(
        default_factory=lambda: SearchComponentSpec("mse")
    )
    direction: Direction = "minimize"
    scale: ObjectiveScale = "scaled"

    def __post_init__(self) -> None:
        if not isinstance(self.component, SearchComponentSpec):
            raise TypeError("component must be a SearchComponentSpec.")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("Invalid objective direction.")
        if self.scale not in ("scaled", "prepared_raw"):
            raise ValueError("Invalid objective scale.")
