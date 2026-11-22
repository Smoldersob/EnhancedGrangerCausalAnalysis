from dataclasses import dataclass
from typing import Literal

from .objective import ObjectiveValue
from .space import SearchCandidate
from .specification import ObjectiveSpec


TrialStatus = Literal["completed", "failed", "invalid"]


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    candidate: SearchCandidate
    status: TrialStatus
    objective: ObjectiveValue | None = None
    seed: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trial_id, str) or not self.trial_id.strip():
            raise ValueError("trial_id must be a non-empty string.")
        if self.status not in ("completed", "failed", "invalid"):
            raise ValueError("Invalid trial status.")
        if self.status == "completed":
            if self.objective is None or self.error is not None:
                raise ValueError(
                    "A completed trial requires an objective and no error."
                )
        elif self.objective is not None or not self.error:
            raise ValueError(
                "An unsuccessful trial requires an error and no objective."
            )


@dataclass(frozen=True)
class ModelSearchResult:
    objective_spec: ObjectiveSpec
    trials: tuple[TrialResult, ...]
    best_trial_id: str

    def __post_init__(self) -> None:
        trials = tuple(self.trials)
        identifiers = [trial.trial_id for trial in trials]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Trial identifiers must be unique.")
        selected = [trial for trial in trials if trial.trial_id == self.best_trial_id]
        if not selected or selected[0].status != "completed":
            raise ValueError("The best trial must be a completed trial.")
        object.__setattr__(self, "trials", trials)

    @property
    def best_trial(self) -> TrialResult:
        return next(trial for trial in self.trials if trial.trial_id == self.best_trial_id)
