from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .layout import RegularizationLayout


@dataclass(frozen=True, eq=False)
class PreparedRegularizerData:
    """Complete prepared regularizer contract."""

    layout: RegularizationLayout
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, RegularizationLayout):
            raise TypeError("layout must be a RegularizationLayout instance.")
        if not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping.")
        if any(not isinstance(key, str) for key in self.params):
            raise TypeError("Parameter names must be strings.")
        object.__setattr__(
            self,
            "params",
            MappingProxyType(deepcopy(dict(self.params))),
        )
