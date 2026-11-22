from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


RegularizerSelector: TypeAlias = str | type


@dataclass(frozen=True, eq=False)
class RegularizerSpec:
    """Normalized selection of a regularizer and parameters."""

    selector: RegularizerSelector
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selector = self.selector
        if isinstance(selector, str):
            selector = selector.strip()
            if not selector:
                raise ValueError("Regularizer name must not be empty.")
        elif not isinstance(selector, type):
            raise TypeError("Regularizer selector must be a name or a class.")

        if not isinstance(self.params, Mapping):
            raise TypeError("Regularizer params must be a mapping.")
        if any(not isinstance(key, str) for key in self.params):
            raise TypeError("Regularizer parameter names must be strings.")

        object.__setattr__(self, "selector", selector)
        object.__setattr__(
            self,
            "params",
            MappingProxyType(deepcopy(dict(self.params))),
        )
