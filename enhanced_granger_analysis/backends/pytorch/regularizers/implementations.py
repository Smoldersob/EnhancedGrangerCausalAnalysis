from __future__ import annotations

import torch

from ....core.components.regularizer_parameters import (
    L1_DEFAULT_PARAMS,
    LAG_L1_DEFAULT_PARAMS,
    prepare_l1_data,
    prepare_lag_l1_data,
)
from ....core.components.regularizer_spec import (
    PreparedRegularizerData,
)
from .base import BasePyTorchRegularizer
from .registry import register_regularizer


@register_regularizer("l1")
class L1Regularizer(BasePyTorchRegularizer):
    DEFAULT_PARAMS = L1_DEFAULT_PARAMS

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        super().__init__(data=prepare_l1_data(data))
        self.l1 = self.data.params["l1"]

    def forward(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._validate_weights(weights)

        return self.l1 * torch.sum(torch.abs(weights))


@register_regularizer("lag_dependent_l1")
class LagDependentL1Regularizer(BasePyTorchRegularizer):
    DEFAULT_PARAMS = LAG_L1_DEFAULT_PARAMS

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        effective_data, indices, feature_weights, origin = (
            prepare_lag_l1_data(data)
        )

        super().__init__(data=effective_data)

        self.l1 = self.data.params["l1"]
        self.relative_lag_indexing = self.data.params[
            "relative_lag_indexing"
        ]
        self.lag_origin = origin

        self.register_buffer(
            "_lag_weight_indices",
            torch.tensor(indices, dtype=torch.int64),
            persistent=False,
        )

        self.register_buffer(
            "_feature_weights",
            torch.tensor(feature_weights, dtype=torch.float64),
            persistent=False,
        )

    @property
    def lag_weight_indices(self) -> torch.Tensor:
        return self._lag_weight_indices.clone()

    @property
    def feature_weights(self) -> torch.Tensor:
        return self._feature_weights.clone()

    def forward(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = self._validate_weights(weights)

        feature_weights = self._feature_weights.to(
            device=weights.device,
            dtype=weights.dtype,
        ).unsqueeze(0)

        return self.l1 * torch.sum(
            torch.abs(weights) * feature_weights
        )