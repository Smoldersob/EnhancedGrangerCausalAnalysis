from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import keras
import numpy as np

from ....core.components.constraint_spec import PreparedConstraintData
from ....core.components.constraint_serialization import (
    constraint_data_from_config,
    constraint_data_to_config,
    copy_constraint_data,
)


class BaseKerasConstraint(keras.constraints.Constraint, ABC):
    """Base constraint for Dense kernels.

    Public mask:
        (n_outputs, n_features)

    Kernel and internal mask:
        (n_features, n_outputs)
    """

    def __init__(
        self,
        data: PreparedConstraintData,
    ) -> None:
        super().__init__()

        self._data = copy_constraint_data(data)

        self.mask = np.array(
            self._data.mask,
            dtype=np.float64,
            copy=True,
        )
        self.rules = self._data.rules

        self._kernel_shape = (
            self._data.n_features,
            len(self._data.output_names),
        )
        self._mask_t = keras.ops.convert_to_tensor(
            self.mask.T,
            dtype="bool",
        )

    @property
    def prepared_data(self) -> PreparedConstraintData:
        return self._data

    def __call__(self, params: Any) -> Any:
        return self.enforce(params)

    @abstractmethod
    def enforce(self, params: Any) -> Any:
        """Return constrained weights using backend tensor operations."""
        ...

    @abstractmethod
    def is_satisfied(self, params: Any) -> bool:
        """Check concrete values in eager mode, outside traced training."""
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