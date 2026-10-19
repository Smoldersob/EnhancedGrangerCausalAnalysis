from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from ....core.components.regularizer_serialization import (
    deserialize_regularizer_data,
    serialize_regularizer_data,
)
from ....core.components.regularizer_spec import (
    PreparedRegularizerData,
)


class BasePyTorchRegularizer(nn.Module, ABC):
    """Regularize matrices shaped (n_outputs, n_features)."""

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        super().__init__()

        if not isinstance(data, PreparedRegularizerData):
            raise TypeError("data must be PreparedRegularizerData.")

        self.data = data

        self.register_buffer(
            "_base_mask",
            torch.tensor(
                data.layout.base_mask.copy(),
                dtype=torch.bool,
            ),
            persistent=False,
        )

    def _validate_weights(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor.")

        if not weights.is_floating_point():
            raise TypeError("weights must have a floating dtype.")

        expected = self.data.layout.coefficient_shape

        if tuple(weights.shape) != expected:
            raise ValueError(
                f"Expected coefficient shape {expected}, "
                f"got {tuple(weights.shape)}."
            )

        return weights

    def base_mask_like(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._base_mask.to(
            device=weights.device,
            dtype=weights.dtype,
        )

    @abstractmethod
    def forward(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
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
    ) -> BasePyTorchRegularizer:
        if not isinstance(config, dict) or set(config) != {"data"}:
            raise ValueError(
                "Config must contain exactly the 'data' field."
            )

        return cls(
            data=deserialize_regularizer_data(config["data"]),
        )