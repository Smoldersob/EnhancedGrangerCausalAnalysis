"""Backend-independent contracts for model search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

Direction = Literal["minimize", "maximize"]
ObjectiveScale = Literal["scaled", "prepared_raw"]
TrialStatus = Literal["completed", "failed", "invalid"]
ParameterPath = tuple[str, ...]


@dataclass(frozen=True)
class ObjectiveValue:
    """Finite aggregate score and optional scores for individual outputs."""

    score: float
    per_output: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        score = float(self.score)
        values = tuple(float(value) for value in self.per_output)

        if not all(isfinite(value) for value in (score, *values)):
            raise ValueError("Objective values must be finite.")

        object.__setattr__(self, "score", score)
        object.__setattr__(self, "per_output", values)


class QualityObjective(Protocol):
    """Evaluate aligned targets and predictions in the supplied scale."""

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ObjectiveValue:
        ...


@dataclass(frozen=True)
class SearchComponentSpec:
    """Component selection without construction or resolution."""

    selector: Any
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise TypeError("Component params must be a mapping.")
        if not all(isinstance(key, str) for key in self.params):
            raise TypeError("Component parameter names must be strings.")

        object.__setattr__(
            self, "params", MappingProxyType(dict(self.params))
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    """Objective declaration and evaluation policy."""

    component: SearchComponentSpec = field(
        default_factory=lambda: SearchComponentSpec("mse")
    )
    direction: Direction = "minimize"
    scale: ObjectiveScale = "scaled"

    def __post_init__(self) -> None:
        if not isinstance(self.component, SearchComponentSpec):
            raise TypeError("component must be a SearchComponentSpec.")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("Invalid objective direction.")
        if self.scale not in ("scaled", "prepared_raw"):
            raise ValueError("Invalid objective scale.")


class SearchDomain:
    """Base contract for normalized parameter domains."""


@dataclass(frozen=True)
class ChoiceDomain(SearchDomain):
    """Explicit alternatives; each entry is one complete value."""

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if not values:
            raise ValueError("A choice domain must contain a value.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SearchParameter:
    """Parameter destination and normalized search domain."""

    path: ParameterPath
    domain: SearchDomain

    def __post_init__(self) -> None:
        path = tuple(self.path)
        if (
            len(path) < 2
            or not all(
                isinstance(item, str) and item and item == item.strip()
                for item in path
            )
        ):
            raise ValueError("A parameter requires a valid section and path.")
        if not isinstance(self.domain, SearchDomain):
            raise TypeError("domain must implement SearchDomain.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class SearchSpace:
    """Normalized parameters without candidate generation."""

    parameters: tuple[SearchParameter, ...]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        paths = [parameter.path for parameter in parameters]

        for index, path in enumerate(paths):
            for other in paths[:index]:
                prefix = min(len(path), len(other))
                if path[:prefix] == other[:prefix]:
                    raise ValueError(
                        f"Conflicting parameter paths: {other!r}, {path!r}."
                    )

        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class CandidateUpdate:
    """One complete replacement at a configuration path."""

    path: ParameterPath
    value: Any

    def __post_init__(self) -> None:
        path = tuple(self.path)
        if (
            len(path) < 2
            or not all(
                isinstance(item, str) and item and item == item.strip()
                for item in path
            )
        ):
            raise ValueError("Invalid candidate update path.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class SearchCandidate:
    """Configuration updates selected by a search strategy."""

    updates: tuple[CandidateUpdate, ...]

    def __post_init__(self) -> None:
        updates = tuple(self.updates)
        SearchSpace(tuple(
            SearchParameter(update.path, ChoiceDomain((update.value,)))
            for update in updates
        ))
        object.__setattr__(self, "updates", updates)


@dataclass(frozen=True)
class TrialResult:
    """Outcome of one candidate evaluation."""

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
    """Search history and reference to the selected successful trial."""

    objective_spec: ObjectiveSpec
    trials: tuple[TrialResult, ...]
    best_trial_id: str

    def __post_init__(self) -> None:
        trials = tuple(self.trials)
        identifiers = [trial.trial_id for trial in trials]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Trial identifiers must be unique.")

        selected = [
            trial for trial in trials
            if trial.trial_id == self.best_trial_id
        ]
        if not selected or selected[0].status != "completed":
            raise ValueError("The best trial must be a completed trial.")

        object.__setattr__(self, "trials", trials)

    @property
    def best_trial(self) -> TrialResult:
        return next(
            trial for trial in self.trials
            if trial.trial_id == self.best_trial_id
        )