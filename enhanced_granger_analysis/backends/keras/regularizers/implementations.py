from __future__ import annotations

from typing import Any

import keras
import numpy as np

from ....core.components.regularizer_parameters import (
    L1_DEFAULT_PARAMS,
    LAG_L1_DEFAULT_PARAMS,
    prepare_l1_data,
    prepare_lag_l1_data,
)
from ....core.components.regularizer_spec import (
    PreparedRegularizerData,
)
from .base import BaseKerasRegularizer
from .registry import register_regularizer


@keras.saving.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="LayoutL1Regularizer",
)
@register_regularizer("l1")
class L1Regularizer(BaseKerasRegularizer):
    DEFAULT_PARAMS = L1_DEFAULT_PARAMS

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        super().__init__(data=prepare_l1_data(data))
        self.l1 = self.data.params["l1"]

    def __call__(self, weights: Any) -> Any:
        weights = self._validate_kernel(weights)

        l1 = keras.ops.convert_to_tensor(
            self.l1,
            dtype=weights.dtype,
        )

        return l1 * keras.ops.sum(keras.ops.abs(weights))


@keras.saving.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="LayoutLagDependentL1Regularizer",
)
@register_regularizer("lag_dependent_l1")
class LagDependentL1Regularizer(BaseKerasRegularizer):
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

        # Keep backend-neutral constants outside traced computations.
        self._lag_weight_indices = indices.copy()
        self._feature_weights = feature_weights.copy()

        self._lag_weight_indices.setflags(write=False)
        self._feature_weights.setflags(write=False)

    @property
    def lag_weight_indices(self) -> np.ndarray:
        return self._lag_weight_indices.copy()

    @property
    def feature_weights(self) -> np.ndarray:
        return self._feature_weights.copy()

    def __call__(self, weights: Any) -> Any:
        weights = self._validate_kernel(weights)

        feature_weights = keras.ops.convert_to_tensor(
            self._feature_weights,
            dtype=weights.dtype,
        )
        feature_weights = keras.ops.reshape(
            feature_weights,
            (self.data.layout.n_features, 1),
        )

        l1 = keras.ops.convert_to_tensor(
            self.l1,
            dtype=weights.dtype,
        )

        return l1 * keras.ops.sum(
            keras.ops.abs(weights) * feature_weights
        )