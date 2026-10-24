from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from ....core.components.regularizer_serialization import (
    deserialize_regularizer_data,
    serialize_regularizer_data,
)
from ....core.components.regularizer_spec import (
    PreparedRegularizerData,
)



class BaseNumpyRegularizer(ABC):
    """
    Regularizer for canonical coefficient matrices:

        (n_outputs, n_features)

    Masking belongs to the model, not to this base class.
    """

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        if not isinstance(data, PreparedRegularizerData):
            raise TypeError("data must be PreparedRegularizerData.")

        effective_params = self._normalize_params(data.params)

        self.data = PreparedRegularizerData(
            layout=data.layout,
            params=effective_params,
        )

    def _normalize_params(
        self,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Custom regularizers may override parameter normalization."""
        return dict(params)

    def _validate_weights(self, weights: Any) -> np.ndarray:
        array = np.asarray(weights)

        if array.shape != self.data.layout.coefficient_shape:
            raise ValueError(
                f"Expected coefficient shape "
                f"{self.data.layout.coefficient_shape}, "
                f"got {array.shape}."
            )

        if array.dtype.kind not in "iuf":
            raise TypeError("Weights must contain real numeric values.")

        if not np.all(np.isfinite(array)):
            raise ValueError("Weights must contain finite values.")

        return array.astype(np.float64, copy=False)

    def __call__(self, weights: Any) -> float:
        penalty, _ = self.value_and_grad(weights)
        return penalty

    def value_and_grad(
        self,
        weights: Any,
    ) -> tuple[float, np.ndarray]:
        array = self._validate_weights(weights)

        penalty, gradient = self._value_and_grad(array)

        penalty_array = np.asarray(penalty)
        gradient_array = np.asarray(gradient)

        if (
            penalty_array.shape != ()
            or penalty_array.dtype.kind not in "iuf"
            or not np.isfinite(penalty_array).item()
        ):
            raise ValueError(
                "Regularizer penalty must be a finite real scalar."
            )

        if gradient_array.shape != array.shape:
            raise ValueError(
                "Regularizer gradient must match the coefficient shape."
            )

        if (
            gradient_array.dtype.kind not in "iuf"
            or not np.all(np.isfinite(gradient_array))
        ):
            raise ValueError(
                "Regularizer gradient must contain finite real values."
            )

        return (
            float(penalty_array),
            gradient_array.astype(np.float64, copy=False),
        )

    @abstractmethod
    def _value_and_grad(
        self,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Compute penalty and its gradient with respect to weights."""
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        return {
            "data": serialize_regularizer_data(self.data),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> BaseNumpyRegularizer:
        if not isinstance(config, Mapping) or set(config) != {"data"}:
            raise ValueError(
                "Regularizer config must contain exactly 'data'."
            )

        return cls(
            data=deserialize_regularizer_data(config["data"]),
        )