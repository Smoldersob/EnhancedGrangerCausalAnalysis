from __future__ import annotations

import warnings

import numpy as np

from ..core.initialization import (
    InitializationContext,
    WeightArray,
)
from .registry import register_initializer


# ------------------------------------------------------------------
# ZEROS INITIALIZER
# ------------------------------------------------------------------
@register_initializer(
    "zeros",
    "zero",
    "zerosinitializer",
)
class ZerosInitializer():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, context):
        """
        Initialize weigh matices with zeros.
        """
        return [
            np.zeros_like(weight)
            for weight in context.weights
        ]

# ------------------------------------------------------------------
# RANDOM NORMAL INITIALIZER (KERAS-STYLE)
# ------------------------------------------------------------------
@register_initializer(
    "random_normal",
    "randomnormal",
    "random",
    "randomnormalinitializer",
)
class RandomNormalInitializer:
    def __init__(
        self,
        *,
        mean: float = 0.0,
        std: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.mean = float(mean)
        self.std = float(std)

        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite.")

        if not np.isfinite(self.std) or self.std < 0:
            raise ValueError("std must be finite and non-negative.")

        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        context: InitializationContext,
    ) -> list[WeightArray]:
        weights = []

        for template in context.weights:
            template = np.asarray(template)

            if not np.issubdtype(template.dtype, np.floating):
                raise TypeError(
                    "RandomNormalInitializer supports only "
                    f"floating-point weights, got {template.dtype}."
                )

            weight = self._rng.normal(
                loc=self.mean,
                scale=self.std,
                size=template.shape,
            )

            weights.append(
                weight.astype(template.dtype, copy=False)
            )

        return weights

# ------------------------------------------------------------------
# OLS-BASED INITIALIZER
# ------------------------------------------------------------------
@register_initializer(
    "ols",
    "ols_initializer",
    "olsinitializer",
    "ordinary_least_squares",
)
class OLSInitializer:
    def __init__(
        self,
        *,
        rcond: float | None = None,
    ) -> None:
        self.rcond = rcond

    def __call__(
        self,
        context: InitializationContext,
    ) -> list[WeightArray]:
        if len(context.weights) > 2:
            warnings.warn(
                "OLSInitializer returns exactly two weight arrays "
                "[W, b], but the model exposes "
                f"{len(context.weights)} arrays. "
                "The returned weights may be incompatible with "
                "the model; validation is delegated to set_weights().",
                UserWarning,
                stacklevel=2,
            )

        X = np.asarray(context.X, dtype=float)
        y = np.asarray(context.y, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be two-dimensional, got shape {X.shape}."
            )

        # Accept a one-dimensional target as a single-output problem.
        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim != 2:
            raise ValueError(
                f"y must be one- or two-dimensional, got shape {y.shape}."
            )

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if n_samples == 0:
            raise ValueError("OLS initialization requires samples.")

        if n_targets == 0:
            raise ValueError(
                "OLS initialization requires at least one target."
            )

        if y.shape[0] != n_samples:
            raise ValueError(
                "X and y must have the same number of samples."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values.")

        mask = None

        if context.mask is not None:
            raw_mask = np.asarray(context.mask)
            expected_mask_shape = (n_targets, n_features)

            if raw_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"mask has shape {raw_mask.shape}; "
                    f"expected {expected_mask_shape}."
                )

            if not np.all((raw_mask == 0) | (raw_mask == 1)):
                raise ValueError(
                    "mask must contain only boolean or binary values."
                )

            mask = raw_mask.astype(bool, copy=False)

        W = np.zeros((n_features, n_targets), dtype=float)
        b = np.zeros(n_targets, dtype=float)

        for target_index in range(n_targets):
            active = (
                np.ones(n_features, dtype=bool)
                if mask is None
                else mask[target_index]
            )

            if not np.any(active):
                b[target_index] = y[:, target_index].mean()
                continue

            X_active = X[:, active]

            design = np.column_stack((
                np.ones(n_samples, dtype=float),
                X_active,
            ))

            coefficients, _, _, _ = np.linalg.lstsq(
                design,
                y[:, target_index],
                rcond=self.rcond,
            )

            b[target_index] = coefficients[0]
            W[active, target_index] = coefficients[1:]

        return [W, b]
