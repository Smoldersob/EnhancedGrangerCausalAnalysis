from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol

import numpy as np


Direction = Literal["minimize", "maximize"]
ObjectiveScale = Literal["scaled", "prepared_raw"]


@dataclass(frozen=True)
class ObjectiveValue:
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
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ObjectiveValue: ...
