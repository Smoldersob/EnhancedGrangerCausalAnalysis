"""Built-in prediction-error objectives and a callable adapter."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any
from numbers import Real

import numpy as np

from .base import (
    BaseObjective,
    Direction,
    ObjectiveResult,
)
from .registry import register_objective


def _validate_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned, finite multi-output observations."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)

    if (
        truth.ndim != 2
        or predictions.ndim != 2
        or truth.shape != predictions.shape
        or truth.shape[0] == 0
        or truth.shape[1] == 0
    ):
        raise ValueError(
            "Targets and predictions must be non-empty, "
            "aligned 2D arrays."
        )

    if truth.dtype.kind not in "iuf" or predictions.dtype.kind not in "iuf":
        raise TypeError(
            "Targets and predictions must contain real numeric values."
        )

    if (
        len(output_names) != truth.shape[1]
        or any(
            not isinstance(name, str) or not name
            for name in output_names
        )
        or len(set(output_names)) != len(output_names)
    ):
        raise ValueError(
            "Output names must be non-empty, unique strings "
            "matching target columns."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        truth = np.asarray(truth, dtype=float)
        predictions = np.asarray(predictions, dtype=float)

    if not (
        np.isfinite(truth).all()
        and np.isfinite(predictions).all()
    ):
        raise ValueError(
            "Targets and predictions must contain finite values."
        )

    return truth, predictions


class _PredictionErrorObjective(BaseObjective):
    """Aggregate per-output errors using optional named weights."""

    def __init__(
        self,
        *,
        output_weights: Mapping[str, float] | None = None,
    ) -> None:
        if output_weights is not None and not isinstance(
            output_weights, Mapping
        ):
            raise TypeError(
                "output_weights must be a mapping or None."
            )

        self._output_weights: dict[str, float] | None = None

        if output_weights is not None:
            weights = {}

            for name, value in output_weights.items():
                if not isinstance(name, str) or not name:
                    raise TypeError(
                        "Output weight names must be non-empty strings."
                    )

                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                ):
                    raise TypeError(
                        "Output weights must be real numeric scalars."
                    )

                weight = float(value)

                if not np.isfinite(weight) or weight < 0:
                    raise ValueError(
                        "Output weights must be finite and non-negative."
                    )

                weights[name] = weight

            if not weights or not any(value > 0 for value in weights.values()):
                raise ValueError(
                    "At least one output weight must be positive."
                )

            self._output_weights = weights

    @property
    def direction(self) -> Direction:
        return "minimize"

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        output_names: tuple[str, ...],
    ) -> ObjectiveResult:
        """Return weighted aggregate and unweighted per-output errors."""
        truth, predictions = _validate_arrays(
            y_true,
            y_pred,
            output_names,
        )

        weights = self._resolve_weights(output_names)

        with np.errstate(over="ignore", invalid="ignore"):
            errors = np.asarray(
                self._per_output(truth, predictions),
                dtype=float,
            )

        if (
            errors.shape != (truth.shape[1],)
            or not np.isfinite(errors).all()
        ):
            raise ValueError(
                "Per-output errors must be finite and aligned."
            )

        return ObjectiveResult(
            score=float(np.dot(weights, errors)),
            scores_by_output=tuple(float(value) for value in errors),
        )

    def _resolve_weights(
        self,
        output_names: tuple[str, ...],
    ) -> np.ndarray:
        if self._output_weights is None:
            return np.full(
                len(output_names),
                1.0 / len(output_names),
                dtype=float,
            )

        if set(self._output_weights) != set(output_names):
            raise ValueError(
                "output_weights must define exactly the model outputs."
            )

        weights = np.asarray(
            [self._output_weights[name] for name in output_names],
            dtype=float,
        )

        weights = weights / weights.max()
        return weights / weights.sum()

    @abstractmethod
    def _per_output(
        self,
        truth: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Return one error value per output."""
        raise NotImplementedError


@register_objective("mse", "mean_squared_error")
class MSEObjective(_PredictionErrorObjective):
    """Compute the weighted mean of per-output MSE values."""

    def _per_output(self, truth, predictions):
        return np.mean((truth - predictions) ** 2, axis=0)


@register_objective("mae", "mean_absolute_error")
class MAEObjective(_PredictionErrorObjective):
    """Compute the weighted mean of per-output MAE values."""

    def _per_output(self, truth, predictions):
        return np.mean(np.abs(truth - predictions), axis=0)


@register_objective("rmse", "root_mean_squared_error")
class RMSEObjective(_PredictionErrorObjective):
    """Compute the weighted mean of per-output RMSE values."""

    def _per_output(self, truth, predictions):
        return np.sqrt(
            np.mean((truth - predictions) ** 2, axis=0)
        )


@register_objective("callable")
class CallableObjective(BaseObjective):
    """Adapt a prediction-scoring function to the objective contract."""

    def __init__(
        self,
        function: Callable[
            [np.ndarray, np.ndarray],
            float | ObjectiveResult,
        ],
        *,
        direction: Direction,
    ) -> None:
        if not callable(function):
            raise TypeError("function must be callable.")

        if direction not in {"minimize", "maximize"}:
            raise ValueError(
                "direction must be 'minimize' or 'maximize'."
            )

        self._function = function
        self._direction = direction

    @property
    def direction(self) -> Direction:
        return self._direction

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        output_names: tuple[str, ...],
    ) -> ObjectiveResult:
        """Evaluate aligned arrays without exposing caller-owned data."""
        truth = np.array(y_true, dtype=float, copy=True)
        predictions = np.array(y_pred, dtype=float, copy=True)

        if (
            truth.ndim != 2
            or predictions.ndim != 2
            or truth.shape != predictions.shape
            or truth.shape[0] == 0
            or truth.shape[1] == 0
        ):
            raise ValueError(
                "Targets and predictions must be non-empty, "
                "aligned 2D arrays."
            )

        if len(output_names) != truth.shape[1]:
            raise ValueError(
                "Output names do not match the number of columns."
            )

        if not (
            np.isfinite(truth).all()
            and np.isfinite(predictions).all()
        ):
            raise ValueError(
                "Targets and predictions must contain finite values."
            )

        value: Any = self._function(truth, predictions)

        if isinstance(value, ObjectiveResult):
            result = value
        else:
            scalar = np.asarray(value)
            if (
                scalar.ndim != 0
                or scalar.dtype.kind not in {"i", "u", "f"}
            ):
                raise TypeError(
                    "The scoring function must return a numeric scalar "
                    "or ObjectiveResult."
                )

            result = ObjectiveResult(score=float(scalar))

        if (
            result.scores_by_output is not None
            and len(result.scores_by_output) != truth.shape[1]
        ):
            raise ValueError(
                "Output scores do not match the number of columns."
            )

        return result