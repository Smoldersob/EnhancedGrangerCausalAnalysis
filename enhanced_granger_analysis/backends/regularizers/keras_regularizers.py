from __future__ import annotations

import importlib
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ...core.exceptions import BackendNotAvailableError, RegularizerConfigurationError


if find_spec("keras") is not None:
	keras = importlib.import_module("keras")
else:  # pragma: no cover - runtime dependency check
	keras = None

import keras
def _ensure_keras() -> None:
	if keras is None:
		raise BackendNotAvailableError("Keras is required to use keras_regularizers.")

@keras.utils.register_keras_serializable(package="EnhancedGrangerCausalAnalysis", name="KerasL1Regularizer")
class KerasL1Regularizer(keras.regularizers.Regularizer if keras is not None else object):
	"""Standard Keras-compatible L1 regularizer with protocol helper methods."""

	def __init__(self, l1: float = 0.0):
		_ensure_keras()
		if l1 < 0:
			raise RegularizerConfigurationError("l1 must be >= 0")
		self.l1 = float(l1)

	def __call__(self, x: Any) -> Any:
		x_cast = keras.ops.cast(x, 'float32')
		if self.l1 == 0.0:
			return keras.opsconstant(0.0, dtype='float32')
		return keras.ops.cast(self.l1, dtype='float32') * keras.ops.sum(keras.ops.abs(x_cast))

	def get_config(self) -> Dict[str, Any]:
		return {"l1": self.l1}


class KerasLagDependentL1Regularizer(keras.regularizers.Regularizer):
    """Lag-dependent L1 regularizer with per-lag influence weights.

    Preferred mapping uses the lag layout produced by LagSelector/LagEngine:
    - max_lags_per_pred
    - col_offsets

    If a layout is not provided and lag_weights has one value per feature,
    it is interpreted as direct per-feature influence.
    """

    _DEFAULT_INITIAL_VALUES: List[float] = [
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0,
    ]

    def __init__(
        self,
        l1: float = 0.0,
        lag_weights: Optional[Sequence[float]] = None,
        max_lags_per_pred: Optional[Sequence[int]] = None,
        col_offsets: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        if l1 < 0:
            raise RegularizerConfigurationError("l1 must be >= 0")

        self.l1 = float(l1)

        if lag_weights is None:
            lag_weights = self._DEFAULT_INITIAL_VALUES

        if len(lag_weights) == 0:
            raise RegularizerConfigurationError(
                "lag_weights cannot be empty"
            )

        self.lag_weights = [float(v) for v in lag_weights]

        if (max_lags_per_pred is None) != (col_offsets is None):
            raise RegularizerConfigurationError(
                "Provide both max_lags_per_pred and col_offsets, or neither."
            )

        self.max_lags_per_pred = None
        self.col_offsets = None

        if max_lags_per_pred is not None:
            self.set_lag_layout(
                max_lags_per_pred=max_lags_per_pred,
                col_offsets=col_offsets,
            )

    def set_lag_layout(
        self,
        max_lags_per_pred: Sequence[int],
        col_offsets: Sequence[int],
    ) -> None:
        """Assign a lag layout produced by LagSelector/LagEngine."""
        if len(max_lags_per_pred) == 0:
            raise RegularizerConfigurationError(
                "max_lags_per_pred cannot be empty"
            )

        if len(max_lags_per_pred) != len(col_offsets):
            raise RegularizerConfigurationError(
                "max_lags_per_pred and col_offsets must have equal length"
            )

        if any(int(value) < 0 for value in max_lags_per_pred):
            raise RegularizerConfigurationError(
                "All max_lags_per_pred values must be >= 0"
            )

        offsets = [int(value) for value in col_offsets]

        if offsets[0] != 0:
            raise RegularizerConfigurationError(
                "col_offsets must start at 0"
            )

        if any(
            offsets[index] > offsets[index + 1]
            for index in range(len(offsets) - 1)
        ):
            raise RegularizerConfigurationError(
                "col_offsets must be non-decreasing"
            )

        self.max_lags_per_pred = [
            int(value) for value in max_lags_per_pred
        ]
        self.col_offsets = offsets

    def _feature_weights_np(self, n_features: int) -> np.ndarray:
        """Return one non-negative or signed influence weight per feature."""
        if (
            self.max_lags_per_pred is not None
            and self.col_offsets is not None
        ):
            if len(self.max_lags_per_pred) != len(self.col_offsets):
                raise RegularizerConfigurationError(
                    "max_lags_per_pred and col_offsets lengths must match"
                )

            offsets = list(self.col_offsets)
            ends = offsets[1:] + [int(n_features)]

            per_feature = np.empty(n_features, dtype=np.float32)

            for predictor_index, (start, end) in enumerate(
                zip(offsets, ends)
            ):
                if start < 0 or end < start or end > n_features:
                    raise RegularizerConfigurationError(
                        "Invalid col_offsets for current number of features"
                    )

                max_lag = int(
                    self.max_lags_per_pred[predictor_index]
                )
                block_len = end - start

                if block_len == 0:
                    continue

                if block_len > max_lag + 1:
                    raise RegularizerConfigurationError(
                        f"Predictor block {predictor_index} has length "
                        f"{block_len}, expected <= {max_lag + 1} "
                        "from max_lags_per_pred"
                    )

                lag_start_idx = max_lag - block_len + 1
                if lag_start_idx < 0:
                    raise RegularizerConfigurationError(
                        f"Cannot infer lag start for block "
                        f"{predictor_index}: max_lag={max_lag}, "
                        f"block_len={block_len}"
                    )

                for rel_idx in range(block_len):
                    lag_idx = lag_start_idx + rel_idx

                    if lag_idx >= len(self.lag_weights):
                        raise RegularizerConfigurationError(
                            f"lag_weights length ({len(self.lag_weights)}) "
                            f"is too short for lag index {lag_idx}"
                        )

                    per_feature[start + rel_idx] = (
                        self.lag_weights[lag_idx]
                    )

            return per_feature

        if len(self.lag_weights) == n_features:
            return np.asarray(self.lag_weights, dtype=np.float32)

        if len(self.lag_weights) == 1:
            return np.full(
                n_features,
                self.lag_weights[0],
                dtype=np.float32,
            )

        raise RegularizerConfigurationError(
            "Unable to map lag_weights to features. "
            "Set max_lags_per_pred and col_offsets, or provide "
            "lag_weights per feature."
        )

    def __call__(self, x: Any) -> Any:
        x_cast = keras.ops.cast(x, "float32")

        if self.l1 == 0.0:
            return keras.ops.convert_to_tensor(0.0, dtype=x_cast.dtype)

        shape = tuple(x_cast.shape)

        if len(shape) == 0:
            l1 = keras.ops.convert_to_tensor(self.l1, dtype=x_cast.dtype)
            return l1 * keras.ops.sum(keras.ops.absolute(x_cast))

        n_features = shape[0]
        if n_features is None:
            raise RegularizerConfigurationError(
                "Kernel first dimension must be statically known for "
                "LagDependentL1Regularizer."
            )

        n_features = int(n_features)

        feature_weights_np = self._feature_weights_np(n_features)
        feature_weights = keras.ops.convert_to_tensor(
            feature_weights_np,
            dtype=x_cast.dtype,
        )

        broadcast_shape = (n_features,) + (1,) * (len(shape) - 1)
        feature_weights = keras.ops.reshape(
            feature_weights,
            broadcast_shape,
        )

        weighted_abs = keras.ops.absolute(x_cast) * feature_weights
        l1 = keras.ops.convert_to_tensor(self.l1, dtype=x_cast.dtype)

        return l1 * keras.ops.sum(weighted_abs)

    def get_params(self) -> Dict[str, Any]:
        """Protocol-compatible helper for regularizer parameters."""
        return {
            "l1": self.l1,
            "lag_weights": list(self.lag_weights),
            "max_lags_per_pred": (
                list(self.max_lags_per_pred)
                if self.max_lags_per_pred is not None
                else None
            ),
            "col_offsets": (
                list(self.col_offsets)
                if self.col_offsets is not None
                else None
            ),
        }

    def get_config(self) -> Dict[str, Any]:
        return self.get_params()

