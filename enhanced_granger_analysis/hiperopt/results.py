from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from collections.abc import Mapping
from typing import Literal
from numbers import Integral

from ..configuration.modeling_spec import ModelingSpec
from ..configuration.model_search_spec import SearchBudget

from ..core.models.training_result import TrainingResult
from .objectives.base import (
    Direction,
    ObjectiveResult,
    ObjectiveScale,
)
from .space.base import SearchCandidate


TrialStatus = Literal["completed", "invalid", "failed"]


@dataclass(frozen=True)
class TrialError:
    """Serializable error information without a traceback object."""

    type_name: str
    message: str


@dataclass(frozen=True)
class TrialResult:
    """Artifacts and evaluation of one search candidate."""

    trial_id: int
    candidate: SearchCandidate
    status: TrialStatus
    objective: ObjectiveResult | None = None
    training_result: TrainingResult | None = None
    training_seed: int | None = None
    duration_seconds: float = 0.0
    error: TrialError | None = None
    epochs_limit: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.trial_id, bool)
            or not isinstance(self.trial_id, int)
            or self.trial_id < 1
        ):
            raise ValueError("trial_id must be a positive integer.")

        if self.status not in {"completed", "invalid", "failed"}:
            raise ValueError("Unsupported trial status.")

        if (
            not isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("Trial duration must be finite and non-negative.")

        if self.status == "completed":
            if self.objective is None or self.error is not None:
                raise ValueError(
                    "A completed trial requires a score and no error."
                )
        elif self.objective is not None or self.error is None:
            raise ValueError(
                "An unsuccessful trial requires an error and no score."
            )

        if self.epochs_limit is not None and (
            isinstance(self.epochs_limit, bool)
            or not isinstance(self.epochs_limit, Integral)
            or self.epochs_limit < 1
        ):
            raise ValueError(
                "epochs_limit must be a positive integer or None."
            )

        if self.training_seed is not None and (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, Integral)
            or self.training_seed < 0
        ):
            raise ValueError(
                "training_seed must be a non-negative integer or None."
            )


@dataclass(frozen=True)
class ModelSearchResult:
    """Selected modeling settings and the complete search history."""

    trials: tuple[TrialResult, ...]
    best_trial_id: int
    best_modeling: ModelingSpec
    direction: Direction
    objective_scale: ObjectiveScale
    validation_part: str
    budget: SearchBudget
    split_parts: Mapping[str, tuple[int, ...]]
    stop_reason: Literal[
        "budget_reached",
        "space_exhausted",
    ]
    duration_seconds: float

    def __post_init__(self) -> None:
        ids = [trial.trial_id for trial in self.trials]

        if len(ids) != len(set(ids)):
            raise ValueError("Trial IDs must be unique.")

        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("Unsupported optimization direction.")

        if self.objective_scale not in {"scaled", "prepared_raw"}:
            raise ValueError("Unsupported objective scale.")

        if self.stop_reason not in {
            "budget_reached",
            "space_exhausted",
        }:
            raise ValueError("Unsupported search stop reason.")

        if (
            not isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError(
                "Search duration must be finite and non-negative."
            )

        if not any(
            trial.trial_id == self.best_trial_id
            and trial.status == "completed"
            for trial in self.trials
        ):
            raise ValueError(
                "The best trial must reference a completed trial."
            )

    @property
    def best_trial(self) -> TrialResult:
        return next(
            trial
            for trial in self.trials
            if trial.trial_id == self.best_trial_id
        )

    @property
    def best_candidate(self) -> SearchCandidate:
        return self.best_trial.candidate

    @property
    def best_score(self) -> float:
        objective = self.best_trial.objective
        assert objective is not None
        return objective.score

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_completed(self) -> int:
        return sum(
            trial.status == "completed"
            for trial in self.trials
        )


class ModelSearchFailedError(RuntimeError):
    """Expose search history when no candidate completed successfully."""

    def __init__(
        self,
        message: str,
        *,
        trials: tuple[TrialResult, ...],
    ) -> None:
        super().__init__(message)
        self.trials = trials