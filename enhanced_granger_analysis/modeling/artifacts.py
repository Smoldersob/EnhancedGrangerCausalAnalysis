from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.models.training_result import TrainingResult


@dataclass
class FitOutcome:
    """Artifacts captured after one successful model fit.

    Predictions use the model's output scale. Weights contain the full
    representation returned by the model, not only coefficient weights.

    Training diagnostics remain in TrainingResult. This object does not
    evaluate predictions or represent a Granger analysis result.
    """

    predictions: np.ndarray
    weights: list[np.ndarray]
    training_result: TrainingResult
    timings: dict[str, float] = field(default_factory=dict)