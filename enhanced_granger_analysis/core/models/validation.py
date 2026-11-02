from __future__ import annotations

from typing import Any

import numpy as np

from .input_spec import ModelInputSpec


def validate_training_arrays(
    X: Any,
    y: Any,
    *,
    input_spec: ModelInputSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy prepared training arrays to float32."""

    def convert(value: Any, name: str) -> np.ndarray:
        raw = np.asarray(value)

        if raw.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if raw.dtype.kind not in "fiu":
            raise TypeError(f"{name} must contain real numeric values.")

        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=np.float32, copy=True)

        if not np.isfinite(result).all():
            raise ValueError(
                f"{name} must contain finite float32-representable values."
            )

        return result

    x_array = convert(X, "X")
    y_array = convert(y, "y")

    if x_array.shape[0] == 0:
        raise ValueError("Training data must contain at least one sample.")

    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have equal sample counts.")

    if x_array.shape[1] != input_spec.n_features:
        raise ValueError(
            f"Expected {input_spec.n_features} input features, "
            f"got {x_array.shape[1]}."
        )

    if y_array.shape[1] != input_spec.n_outputs:
        raise ValueError(
            f"Expected {input_spec.n_outputs} outputs, "
            f"got {y_array.shape[1]}."
        )

    return x_array, y_array