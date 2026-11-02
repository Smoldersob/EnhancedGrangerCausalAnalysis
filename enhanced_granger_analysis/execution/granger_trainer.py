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
        """Fit the adapter's configured baseline state."""
        return self._fit_model(
            model=model,
            prepared=prepared,
            cause=None,
            initial_weights=initial_weights,
            run_name="base_model",
            restricted=False,
        )

    def fit_unrestricted(
        self, *, model: GrangerModelProtocol, prepared: PreparedData,
        cause: str, base_weights: Sequence[np.ndarray] | None = None,
        initial_weights: Sequence[np.ndarray] | None = None,
    ) -> FitOutcome:
        return self._fit_model(
            model=model, prepared=prepared, cause=cause,
            base_weights=base_weights, initial_weights=initial_weights,
            run_name=f"unrestricted_cause_{tuple(prepared.causes).index(cause)}",
            restricted=False,
        )

    def fit_restricted(
        self, *, model: GrangerModelProtocol, prepared: PreparedData,
        cause: str, base_weights: Sequence[np.ndarray] | None = None,
        initial_weights: Sequence[np.ndarray] | None = None,
    ) -> FitOutcome:
        return self._fit_model(
            model=model, prepared=prepared, cause=cause,
            base_weights=base_weights, initial_weights=initial_weights,
            run_name=f"restricted_cause_{tuple(prepared.causes).index(cause)}",
            restricted=True,
        )

    def _fit_model(
        self, *, model: GrangerModelProtocol, prepared: PreparedData,
        cause: str | None = None,
        base_weights: Sequence[np.ndarray] | None = None,
        initial_weights: Sequence[np.ndarray] | None = None,
        run_name: str = "model",
        restricted: bool,
    ) -> FitOutcome:
        """Fit one explicit configuration without owning result storage."""

        self._require_full_training_sample(prepared)
        if cause is not None and cause not in prepared.cause_slices:
            raise DataValidationError(
                f"Unknown cause: {cause!r}."
            )

        timings: dict[str, float] = {}
        started = perf_counter()
        if base_weights is not None:
            self._restore_weights(model, base_weights)
        elif initial_weights is not None:
            self._restore_weights(model, initial_weights)

        timings["restore_weights"] = perf_counter() - started
        started = perf_counter()
        indices = [] if cause is None else list(range(
            prepared.cause_slices[cause].start,
            prepared.cause_slices[cause].stop,
        ))
        if restricted:
            model.omit_variables(indices)
        else:
            model.add_variables(indices)
        timings["update_mask"] = perf_counter() - started
        started = perf_counter()
        model.reset_components(
            reset_optimizer=True,
            reset_callbacks=True,
            run_name=run_name,
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

        active_mask_getter = getattr(model, "get_active_mask", None)
        if callable(active_mask_getter):
            active_mask = np.asarray(active_mask_getter(), dtype=bool).copy()
        else:
            structural = getattr(getattr(model, "constraint", None), "mask", None)
            if structural is None:
                active_mask = np.ones(
                    (model.input_spec.n_outputs, model.input_spec.n_features),
                    dtype=bool,
                )
            else:
                active_mask = np.asarray(structural, dtype=bool).copy()
            omitted = np.asarray(getattr(model, "omitted", []), dtype=np.intp)
            if omitted.size:
                active_mask[:, omitted] = False

        return FitOutcome(
            predictions=predictions,
            weights=weights,
            training_result=training_result,
            timings=dict(timings),
            active_slope_mask=active_mask,
            bias_enabled=bool(getattr(model, "use_bias", len(weights) > 1)),
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

    @staticmethod
    def _require_full_training_sample(
        prepared: PreparedData,
    ) -> None:
        """Reject held-out rows in the standard Granger workflow."""
        training = prepared.indices_for(
            "training",
            require_nonempty=True,
        )
        expected = np.arange(
            prepared.X_raw.shape[0],
            dtype=np.intp,
        )

        if not np.array_equal(training, expected):
            raise DataValidationError(
                "The standard Granger workflow currently requires "
                "all prepared rows to belong to training. "
                "Use the split-aware executor for HPO. "
                "Prepare a separate analytical sample for final refitting."
            )