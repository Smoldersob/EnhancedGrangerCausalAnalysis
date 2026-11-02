from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ....core.components.constraint_spec import PreparedConstraintData
from ....core.components.constraint_serialization import (
    constraint_data_from_config,
    constraint_data_to_config,
    copy_constraint_data,
)

if TYPE_CHECKING:
    from torch import Tensor


class BasePyTorchConstraint(ABC):
    """Base constraint for weights shaped (n_outputs, n_features)."""

    def __init__(
        self,
        data: PreparedConstraintData,
    ) -> None:
        self._data = copy_constraint_data(data)

        # Writable copy avoids sharing the read-only prepared mask.
        self.mask = np.array(
            self._data.mask,
            dtype=np.float64,
            copy=True,
        )
        self.rules = self._data.rules

    @property
    def prepared_data(self) -> PreparedConstraintData:
        return self._data

    def __call__(self, params: Tensor) -> Tensor:
        return self.enforce(params)

    @abstractmethod
    def enforce(self, params: Tensor) -> Tensor:
        """Return constrained weights without modifying the input."""
        ...

    @abstractmethod
    def is_satisfied(self, params: Tensor) -> bool:
        ...

    def get_config(self) -> dict[str, Any]:
        """Subclasses must also serialize their own constructor parameters."""
        return {
            "data": constraint_data_to_config(self._data),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        params = dict(config)
        data = constraint_data_from_config(params.pop("data"))
        return cls(data=data, **params)