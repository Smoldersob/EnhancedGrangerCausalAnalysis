# enhanced_granger_analysis/core/execution/session.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..components.constraint_spec import PreparedConstraintData
from ..components.regularizer_spec import PreparedRegularizerData
from ...backends.base_backend import BackendStrategy
from ...preprocessing.prepare_data import PreparedData
from ...results.granger_results import GrangerAnalysisResults

if TYPE_CHECKING:
    from ...api.analysis_config import AnalysisConfig


Stage = Literal["data", "components", "model", "initialization", "fit"]


@dataclass
class AnalysisSession:
    """Store artifacts belonging to one analysis execution."""

    config: AnalysisConfig

    prepared_data: PreparedData | None = None
    strategy: BackendStrategy | None = None

    constraint_data: PreparedConstraintData | None = None
    regularizer_data: PreparedRegularizerData | None = None

    constraint: Any = None
    regularizer: Any = None
    optimizer: Any = None
    loss: Any = None
    callbacks: list[Any] | None = None

    # Replace Any with the project's adapter protocol when integrating.
    model: Any = None

    initial_weights: list[np.ndarray] | None = None
    base_weights: list[np.ndarray] | None = None

    results: GrangerAnalysisResults | None = None
    hyperoptimization_result: Any = None

    completed_stages: set[Stage] = field(default_factory=set)

    @property
    def effective_mask(self) -> np.ndarray:
        """Return the effective constraint mask, falling back to the base mask.

        Raise the project's validation error when data is unavailable. Return
        the canonical session mask for inspection; callers must not mutate it
        in place. Use an explicit copy when passing it to mutable consumers.
        """
        raise NotImplementedError