"""Declarative configuration and resource limits for model search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal

from ..hiperopt.objectives.base import ObjectiveScale
from ..hiperopt.space.base import SearchSpace


ErrorPolicy = Literal["raise", "record"]


def _positive_integer(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class SearchBudget:
    """Limit candidate evaluations and epochs of individual fits."""

    max_trials: int = 20
    epochs_per_trial: int | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.max_trials, "max_trials")

        if self.epochs_per_trial is not None:
            _positive_integer(
                self.epochs_per_trial,
                "epochs_per_trial",
            )


@dataclass(frozen=True)
class ModelSearchSpec:
    """Declare search inputs without constructing components."""

    space: Mapping[str, Any] | SearchSpace
    strategy: Any
    objective: Any
    budget: SearchBudget = field(default_factory=SearchBudget)
    validation_part: str = "validation"
    objective_scale: ObjectiveScale = "scaled"
    error_policy: ErrorPolicy = "raise"
    training_seed: int | None = 42
    collect_history: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.space, (Mapping, SearchSpace)):
            raise TypeError(
                "space must be a declaration mapping or SearchSpace."
            )

        if self.strategy is None or self.objective is None:
            raise ValueError(
                "An active search requires a strategy and objective."
            )

        if not isinstance(self.budget, SearchBudget):
            raise TypeError("budget must be a SearchBudget.")

        if (
            not isinstance(self.validation_part, str)
            or not self.validation_part.strip()
            or self.validation_part != self.validation_part.strip()
            or self.validation_part in {"training", "test"}
        ):
            raise ValueError("Invalid validation part name.")

        if self.objective_scale not in {"scaled", "prepared_raw"}:
            raise ValueError("Unsupported objective scale.")

        if self.error_policy not in {"raise", "record"}:
            raise ValueError("Unsupported search error policy.")

        if self.training_seed is not None and (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, Integral)
            or self.training_seed < 0
        ):
            raise ValueError(
                "training_seed must be a non-negative integer or None."
            )

        if not isinstance(self.collect_history, bool):
            raise TypeError("collect_history must be bool.")