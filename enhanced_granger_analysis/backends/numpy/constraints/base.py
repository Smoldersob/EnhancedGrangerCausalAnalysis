from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ....core.components.constraint_spec import PreparedConstraintData
from ....core.components.constraint_serialization import (
    constraint_data_from_config,
    constraint_data_to_config,
    copy_constraint_data,
)


class BaseNumpyConstraint(ABC):
    """Base constraint for weights shaped (n_outputs, n_features)."""

    def __init__(
        self,
        data: PreparedConstraintData,
    ) -> None:
        self._data = copy_constraint_data(data)

        # Independent, writable backend representation.
        self.mask = np.array(
            self._data.mask,
            dtype=np.float64,
            copy=True,
        )
        self.rules = self._data.rules

    @property
    def prepared_data(self) -> PreparedConstraintData:
        return self._data

    def __call__(
        self,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.enforce(params)

    @abstractmethod
    def enforce(
        self,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return constrained weights without modifying the input."""
        ...

    @abstractmethod
    def is_satisfied(
        self,
        params: NDArray[np.float64],
    ) -> bool:
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

    def zero_only_mask(self) -> np.ndarray | None:
        """Describe this constraint as pure coefficient zeroing.

        Return a boolean mask if it completely describes the
        feasible set:
            False -> coefficient must be zero
            True  -> coefficient is unrestricted

        Return None if this representation is not declared.
        """
        return None