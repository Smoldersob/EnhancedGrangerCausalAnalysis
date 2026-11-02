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
    preparation: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelSearchSpec":
        """Parse a search declaration without resolving runtime components."""
        if not isinstance(config, Mapping):
            raise TypeError("search must be a mapping or None.")

        allowed = {
            "space", "strategy", "objective", "budget", "validation_part",
            "objective_scale", "error_policy", "training_seed",
            "collect_history", "preparation",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                "Unknown model search fields: "
                f"{sorted(unknown, key=repr)!r}."
            )

        if "space" not in config or "strategy" not in config or "objective" not in config:
            raise ValueError(
                "An active search requires space, strategy, and objective."
            )

        budget_value = config.get("budget", {})
        if isinstance(budget_value, SearchBudget):
            budget = budget_value
        else:
            if not isinstance(budget_value, Mapping):
                raise TypeError("search.budget must be a mapping.")
            budget_unknown = set(budget_value) - {
                "max_trials", "epochs_per_trial",
            }
            if budget_unknown:
                raise ValueError(
                    "Unknown search budget fields: "
                    f"{sorted(budget_unknown, key=repr)!r}."
                )
            budget = SearchBudget(**dict(budget_value))

        preparation = config.get("preparation")
        if preparation is not None:
            if not isinstance(preparation, Mapping):
                raise TypeError("search.preparation must be a mapping or None.")
            preparation = dict(preparation)
            invalid = set(preparation) - {"splitter", "random_state"}
            if invalid:
                raise ValueError(
                    "Unsupported search preparation overrides: "
                    f"{sorted(invalid, key=repr)!r}. Only splitter and "
                    "random_state may be overridden."
                )

        values = {
            key: config[key]
            for key in (
                "space", "strategy", "objective", "validation_part",
                "objective_scale", "error_policy", "training_seed",
                "collect_history",
            )
            if key in config
        }
        return cls(**values, budget=budget, preparation=preparation)

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

        if self.preparation is not None:
            if not isinstance(self.preparation, Mapping):
                raise TypeError(
                    "preparation must be a mapping or None."
                )
            unknown = set(self.preparation) - {"splitter", "random_state"}
            if unknown:
                raise ValueError(
                    "Unsupported search preparation overrides: "
                    f"{sorted(unknown, key=repr)!r}."
                )