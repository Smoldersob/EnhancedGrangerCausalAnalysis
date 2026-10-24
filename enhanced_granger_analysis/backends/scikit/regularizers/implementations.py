from __future__ import annotations

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
from .base import BaseScikitRegularizer
from .registry import register_regularizer


@register_regularizer("l1")
class L1Regularizer(BaseScikitRegularizer):
    DEFAULT_PARAMS = L1_DEFAULT_PARAMS

    def __init__(
        self,
        *,
        data: PreparedRegularizerData,
    ) -> None:
        super().__init__(data=prepare_l1_data(data))
        self.l1 = self.data.params["l1"]

    def _value_and_grad(
        self,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        array = self._validate_weights(weights)

        return (
            float(self.l1 * np.sum(np.abs(array))),
            self.l1 * np.sign(array),
        )

    def prox(
        self,
        weights: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        array = self._validate_weights(weights)

        raw_step = np.asarray(step_size)
        if raw_step.shape != () or raw_step.dtype.kind not in "iuf":
            raise TypeError(
                "step_size must be a real numeric scalar."
            )

        step = float(raw_step)
        if not np.isfinite(step) or step < 0:
            raise ValueError(
                "step_size must be finite and non-negative."
            )

        threshold = step * self.l1

        if not np.isfinite(threshold):
            raise ValueError("L1 threshold is non-finite.")

        return np.sign(array) * np.maximum(
            np.abs(array) - threshold,
            0.0,
        )


@register_regularizer("lag_dependent_l1")
class LagDependentL1Regularizer(BaseScikitRegularizer):
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

        self._lag_weights = np.array(
            self.data.params["lag_weights"],
            dtype=np.float64,
        )
        self._lag_weight_indices = indices
        self._feature_weights = feature_weights

        self._lag_weights.setflags(write=False)
        self._lag_weight_indices.setflags(write=False)
        self._feature_weights.setflags(write=False)

    @property
    def lag_weights(self) -> np.ndarray:
        return self._lag_weights.copy()

    @property
    def lag_weight_indices(self) -> np.ndarray:
        return self._lag_weight_indices.copy()

    @property
    def feature_weights(self) -> np.ndarray:
        return self._feature_weights.copy()

    def _value_and_grad(
        self,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        array = self._validate_weights(weights)
        feature_weights = self._feature_weights[np.newaxis, :]

        return (
            float(
                self.l1
                * np.sum(np.abs(array) * feature_weights)
            ),
            self.l1 * np.sign(array) * feature_weights,
        )
    
    def prox(
        self,
        weights: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        array = self._validate_weights(weights)

        raw_step = np.asarray(step_size)
        if raw_step.shape != () or raw_step.dtype.kind not in "iuf":
            raise TypeError(
                "step_size must be a real numeric scalar."
            )

        step = float(raw_step)
        if not np.isfinite(step) or step < 0:
            raise ValueError(
                "step_size must be finite and non-negative."
            )

        threshold = (
            step
            * self.l1
            * self._feature_weights[np.newaxis, :]
        )

        if not np.isfinite(threshold).all():
            raise ValueError("Lag-dependent L1 threshold is non-finite.")

        return np.sign(array) * np.maximum(
            np.abs(array) - threshold,
            0.0,
        )