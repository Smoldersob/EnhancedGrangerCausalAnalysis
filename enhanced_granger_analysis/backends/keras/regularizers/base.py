from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import keras

from ....core.components.regularizer_serialization import (
    deserialize_regularizer_data,
    serialize_regularizer_data,
)
from ....core.components.regularizer_spec import (
    PreparedRegularizerData,
)


class BaseKerasRegularizer(
    keras.regularizers.Regularizer,
    ABC,
):
    """Regularize kernels shaped (n_features, n_outputs)."""

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        super().__init__()

        if not isinstance(data, PreparedRegularizerData):
            raise TypeError("data must be PreparedRegularizerData.")

        self.data = data

    def _validate_kernel(self, weights: Any) -> Any:
        shape = tuple(weights.shape)
        layout = self.data.layout
        expected = (layout.n_features, layout.n_outputs)

        if shape != expected:
            raise ValueError(
                f"Expected kernel shape {expected}, got {shape}."
            )

        return weights

    def to_canonical(self, weights: Any) -> Any:
        weights = self._validate_kernel(weights)
        return keras.ops.transpose(weights, axes=(1, 0))

    @abstractmethod
    def __call__(self, weights: Any) -> Any:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {
            "data": serialize_regularizer_data(self.data),
        }

    def get_params(self) -> dict[str, Any]:
        return self.get_config()["data"]["params"]

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> BaseKerasRegularizer:
        if not isinstance(config, dict) or set(config) != {"data"}:
            raise ValueError(
                "Config must contain exactly the 'data' field."
            )

        return cls(
            data=deserialize_regularizer_data(config["data"]),
        )