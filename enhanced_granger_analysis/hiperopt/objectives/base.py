"""Backend-independent contracts for prediction-quality objectives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np


Direction = Literal["minimize", "maximize"]
ObjectiveScale = Literal["scaled", "prepared_raw"]


def _finite_float(value: Real, name: str) -> float:
    """Validate a finite real scalar and return its float value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar.")

    result = float(value)

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


@dataclass(frozen=True)
class ObjectiveResult:
    """Scalar selection score and optional per-output diagnostics."""

    score: float
    scores_by_output: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "score",
            _finite_float(self.score, "score"),
        )

        if self.scores_by_output is not None:
            values = tuple(
                _finite_float(value, "Per-output score")
                for value in self.scores_by_output
            )

            if not values:
                raise ValueError("scores_by_output cannot be empty.")

            object.__setattr__(self, "scores_by_output", values)


class BaseObjective(ABC):
    """Score aligned predictions without training or preprocessing."""

    @property
    @abstractmethod
    def direction(self) -> Direction:
        """Return whether smaller or larger scores are preferred."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        output_names: tuple[str, ...],
    ) -> ObjectiveResult:
        """Score arrays shaped as samples by outputs."""
        raise NotImplementedError