from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from ..core.exceptions import DataValidationError
from ..core.models.protocol import GrangerModelProtocol
from ..core.models.training_result import TrainingResult
from ..modeling.artifacts import FitOutcome
from ..modeling.executor import FitExecutor
from ..preprocessing.prepare_data import PreparedData


class GrangerTrainer:
    """Execute standard unrestricted and restricted model fits.

    Each operation returns artifacts of one fit. This service does not
    create or modify Granger analysis results.
    """

    def __init__(
        self,
        executor: FitExecutor | None = None,
    ) -> None:
        self._executor = (
            executor if executor is not None else FitExecutor()
        )

    def fit_base(
        self,
        *,
        model: GrangerModelProtocol,
        prepared: PreparedData,
        initial_weights: Sequence[np.ndarray] | None = None,
    ) -> FitOutcome:
        """Fit the unrestricted model and capture its artifacts."""

        timings: dict[str, float] = {}

        started = perf_counter()

        if initial_weights is not None:
            self._restore_weights(model, initial_weights)

        timings["restore_weights"] = perf_counter() - started

        started = perf_counter()

        model.omit_variables([])

        timings["update_mask"] = perf_counter() - started

        started = perf_counter()

        model.reset_components(
            reset_optimizer=True,
            reset_callbacks=True,
            run_name="base_model",
        )

        timings["reset_components"] = perf_counter() - started

        started = perf_counter()

        training_result = self._executor.fit(
            model=model,
            X=prepared.X_scaled,
            y=prepared.y_scaled,
            reset_components=False,
        )

        timings["fit_model"] = perf_counter() - started

        return self._capture_outcome(
            model=model,
            X=prepared.X_scaled,
            training_result=training_result,
            timings=timings,
        )

    def fit_restricted(
        self,
        *,
        model: GrangerModelProtocol,
        prepared: PreparedData,
        cause: str,
        base_weights: Sequence[np.ndarray] | None,
        initial_weights: Sequence[np.ndarray] | None = None,
        start_with_base: bool = True,
        start_with_initial: bool = False,
    ) -> FitOutcome:
        """Fit one restricted model and capture its artifacts."""

        if base_weights is None:
            raise DataValidationError(
                "fit_base() must be completed before fit_restricted()."
            )

        if cause not in prepared.cause_slices:
            raise DataValidationError(
                f"Unknown cause: {cause!r}."
            )

        timings: dict[str, float] = {}

        started = perf_counter()

        if start_with_initial and initial_weights is not None:
            self._restore_weights(model, initial_weights)
        elif start_with_base:
            self._restore_weights(model, base_weights)

        timings["restore_weights"] = perf_counter() - started

        started = perf_counter()

        block = prepared.cause_slices[cause]

        model.omit_variables(
            list(range(block.start, block.stop))
        )

        timings["update_mask"] = perf_counter() - started

        cause_index = tuple(prepared.causes).index(cause)

        started = perf_counter()

        model.reset_components(
            reset_optimizer=True,
            reset_callbacks=True,
            run_name=f"reference_cause_{cause_index}",
        )

        timings["reset_components"] = perf_counter() - started

        started = perf_counter()

        training_result = self._executor.fit(
            model=model,
            X=prepared.X_scaled,
            y=prepared.y_scaled,
            reset_components=False,
        )

        timings["fit_model"] = perf_counter() - started

        return self._capture_outcome(
            model=model,
            X=prepared.X_scaled,
            training_result=training_result,
            timings=timings,
        )

    @staticmethod
    def _capture_outcome(
        *,
        model: GrangerModelProtocol,
        X: np.ndarray,
        training_result: TrainingResult,
        timings: dict[str, float],
    ) -> FitOutcome:
        """Capture independent prediction and weight arrays."""

        predictions = np.array(
            model.predict(X),
            dtype=float,
            copy=True,
        )

        weights = [
            np.asarray(weight).copy()
            for weight in model.get_weights()
        ]

        return FitOutcome(
            predictions=predictions,
            weights=weights,
            training_result=training_result,
            timings=dict(timings),
        )

    @staticmethod
    def _restore_weights(
        model: GrangerModelProtocol,
        weights: Sequence[np.ndarray],
    ) -> None:
        """Restore independent copies through the model contract."""

        model.set_weights([
            np.asarray(weight).copy()
            for weight in weights
        ])