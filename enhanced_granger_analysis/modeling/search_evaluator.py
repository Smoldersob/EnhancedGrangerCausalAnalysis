"""Evaluate one HPO candidate through shared modeling services."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from ..backends import BackendFactory

from ..configuration.modeling_spec import ModelingSpec

from ..core.exceptions import DataValidationError
from ..core.components.regularizer_parsing import (
    parse_regularizer_spec,
)

from ..hiperopt.objectives.base import (
    BaseObjective,
    ObjectiveResult,
    ObjectiveScale,
)
from ..hiperopt.results import TrialError, TrialResult
from ..hiperopt.space.base import SearchCandidate

from ..preprocessing.prepare_data import PreparedData

from .assembler import ModelAssembler
from .executor import FitExecutor
from .initialization import ModelInitializer
from .search_configuration import (
    apply_modeling_candidate,
    apply_trial_epochs,
)

class ModelCandidateEvaluator:
    """Build, fit and score an independent multi-output candidate."""

    def __init__(
        self,
        *,
        assembler: ModelAssembler | None = None,
        initializer: ModelInitializer | None = None,
        executor: FitExecutor | None = None,
    ) -> None:
        self._assembler = (
            assembler if assembler is not None else ModelAssembler()
        )
        self._initializer = (
            initializer if initializer is not None else ModelInitializer()
        )
        self._executor = (
            executor if executor is not None else FitExecutor()
        )

    def evaluate(
        self,
        *,
        trial_id: int,
        candidate: SearchCandidate,
        modeling: ModelingSpec,
        prepared: PreparedData,
        objective: BaseObjective,
        validation_part: str = "validation",
        objective_scale: ObjectiveScale = "scaled",
        training_seed: int | None = None,
        epochs_per_trial: int | None = None,
        error_policy: str = "raise",
        collect_history: bool = False,
    ) -> TrialResult:
        """Evaluate a candidate without modifying prepared observations."""
        self._validate_options(
            trial_id=trial_id,
            validation_part=validation_part,
            objective_scale=objective_scale,
            training_seed=training_seed,
            error_policy=error_policy,
        )

        self.validate_inputs(
            prepared=prepared,
            objective=objective,
            validation_part=validation_part,
            objective_scale=objective_scale,
        )

        training, validation = self._select_parts(
            prepared,
            validation_part,
        )

        started = perf_counter()
        if epochs_per_trial is not None and (
            isinstance(epochs_per_trial, bool)
            or not isinstance(epochs_per_trial, Integral)
            or epochs_per_trial < 1
        ):
            raise ValueError(
                "epochs_per_trial must be a positive integer or None."
            )
        training_result = None
        status = "invalid"

        try:
            candidate_modeling = apply_modeling_candidate(
                modeling,
                candidate,
            )

            trial_modeling = apply_trial_epochs(
                candidate_modeling,
                epochs_per_trial,
            )

            model_config = trial_modeling.model_config
            regularizer_declaration = trial_modeling.regularizer_spec

            self._validate_component_inputs(
                model_config=model_config,
                callbacks=trial_modeling.callbacks,
                constraint_spec=trial_modeling.constraint_spec,
                regularizer_spec=regularizer_declaration,
                initializer_spec=trial_modeling.initializer,
            )

            strategy = BackendFactory.get_strategy(modeling.backend)

            if modeling.constraint_spec is None:
                raise DataValidationError(
                    "constraint_spec cannot be None for HPO."
                )

            constraint_data = self._assembler.prepare_constraint_data(
                prepared=prepared,
                relations=modeling.relations,
            )

            constraint = self._assembler.build_constraint(
                strategy=strategy,
                constraint_spec=modeling.constraint_spec,
                data=constraint_data,
            )

            regularizer_data = self._assembler.prepare_regularizer_data(
                prepared=prepared,
                regularizer_spec=regularizer_declaration,
            )

            regularizer = self._assembler.build_regularizer(
                strategy=strategy,
                regularizer_spec=regularizer_declaration,
                data=regularizer_data,
            )

            assembly = self._assembler.assemble(
                strategy=strategy,
                prepared=prepared,
                model_config=model_config,
                callbacks=modeling.callbacks,
                constraint=constraint,
                regularizer=regularizer,
            )

            model = assembly.model
            if epochs_per_trial is not None:
                actual_limit = getattr(model, "epochs", None)

                if (
                    isinstance(actual_limit, bool)
                    or not isinstance(actual_limit, Integral)
                    or int(actual_limit) != int(epochs_per_trial)
                ):
                    raise DataValidationError(
                        "The selected model does not expose the requested "
                        "trial budget through its epochs attribute."
                    )
            status = "failed"

            if training_seed is not None:
                model.rebuild(seed=int(training_seed))

            model.omit_variables([])

            weights = self._initializer.compute_weights(
                model=model,
                prepared=prepared,
                initializer_spec=modeling.initializer,
                row_indices=training,
                initialization_mask=constraint_data.mask,
            )

            if weights is not None:
                model.set_weights([
                    np.asarray(weight).copy()
                    for weight in weights
                ])

            training_result = self._executor.fit(
                model=model,
                X=prepared.X_scaled[training],
                y=prepared.y_scaled[training],
                collect_history=collect_history,
                reset_components=True,
                run_name=f"hpo_trial_{trial_id}",
            )

            predictions = np.asarray(
                model.predict(prepared.X_scaled[validation]),
                dtype=float,
            )

            expected_shape = prepared.y_scaled[validation].shape
            if predictions.shape != expected_shape:
                raise DataValidationError(
                    "Prediction shape does not match validation targets."
                )

            if not np.isfinite(predictions).all():
                raise DataValidationError(
                    "Validation predictions must be finite."
                )

            if objective_scale == "prepared_raw":
                if prepared.y_scaler is None:
                    raise DataValidationError(
                        "prepared_raw scoring requires a fitted "
                        "y_scaler or an identity scaler."
                    )

                predictions = np.asarray(
                    prepared.y_scaler.inverse_transform(
                        predictions.copy()
                    ),
                    dtype=float,
                )
                targets = prepared.y_raw[validation]
            else:
                targets = prepared.y_scaled[validation]

            if (
                predictions.shape != targets.shape
                or not np.isfinite(predictions).all()
                or not np.isfinite(targets).all()
            ):
                raise DataValidationError(
                    "Scoring requires aligned, finite targets "
                    "and predictions."
                )

            score = objective.evaluate(
                np.array(targets, copy=True),
                np.array(predictions, copy=True),
                output_names=tuple(prepared.effects),
            )

            self._validate_score(score, len(prepared.effects))

            return TrialResult(
                trial_id=trial_id,
                candidate=candidate,
                status="completed",
                objective=score,
                training_result=training_result,
                training_seed=training_seed,
                duration_seconds=perf_counter() - started,
                epochs_limit=epochs_per_trial,
            )

        except Exception as exc:
            if error_policy == "raise":
                raise

            return TrialResult(
                trial_id=trial_id,
                candidate=candidate,
                status=status,
                training_result=training_result,
                training_seed=training_seed,
                duration_seconds=perf_counter() - started,
                error=TrialError(
                    type_name=type(exc).__name__,
                    message=str(exc),
                ),
                epochs_limit=epochs_per_trial,
            )

    @staticmethod
    def _select_parts(
        prepared: PreparedData,
        validation_part: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate prepared arrays and return independent part indices."""
        X = prepared.X_scaled
        y = prepared.y_scaled

        if (
            X.ndim != 2
            or y.ndim != 2
            or X.shape[0] != y.shape[0]
            or X.shape[0] == 0
            or X.shape[1] == 0
            or y.shape[1] == 0
        ):
            raise DataValidationError(
                "HPO requires non-empty, aligned 2D prepared arrays."
            )

        if prepared.y_raw.shape != y.shape:
            raise DataValidationError(
                "Raw and scaled targets must have identical shapes."
            )

        if len(prepared.effects) != y.shape[1]:
            raise DataValidationError(
                "Output names do not match target columns."
            )

        if prepared.split is None:
            raise DataValidationError(
                "HPO requires an explicit split with training "
                "and validation observations."
            )

        collected = []
        selected = {}

        for name, values in prepared.split.parts.items():
            indices = np.asarray(values)

            if (
                indices.ndim != 1
                or indices.dtype.kind not in {"i", "u"}
            ):
                raise DataValidationError(
                    f"Part {name!r} must contain 1D integer indices."
                )

            if (
                np.any(indices < 0)
                or np.any(indices >= len(X))
                or np.any(indices[1:] <= indices[:-1])
            ):
                raise DataValidationError(
                    f"Part {name!r} contains invalid indices."
                )

            selected[name] = indices.astype(np.intp, copy=True)
            collected.append(indices)

        if collected:
            combined = np.concatenate(collected)
            if np.unique(combined).size != combined.size:
                raise DataValidationError(
                    "Data parts must not share prepared rows."
                )

        training = selected.get("training")
        validation = selected.get(validation_part)

        if training is None or training.size == 0:
            raise DataValidationError(
                "HPO requires a non-empty training part."
            )

        if validation is None or validation.size == 0:
            raise DataValidationError(
                f"HPO requires a non-empty {validation_part!r} part."
            )

        for indices in (training, validation):
            if not (
                np.isfinite(X[indices]).all()
                and np.isfinite(y[indices]).all()
            ):
                raise DataValidationError(
                    "Training and validation arrays must be finite."
                )

        return training, validation

    @staticmethod
    def _require_constructible(spec: Any, label: str) -> None:
        """Reject ready components lacking a trial-isolation contract."""
        if spec is None or isinstance(spec, (str, type)):
            return

        if isinstance(spec, Mapping):
            selector = spec.get("type")
            if isinstance(selector, (str, type)):
                return

        raise TypeError(
            f"HPO requires a name, class or declaration for {label}. "
            "Ready instances are not supported by this evaluator."
        )

    @classmethod
    def _validate_component_inputs(
        cls,
        *,
        model_config: Mapping[str, Any],
        callbacks: Any,
        constraint_spec: Any,
        regularizer_spec: Any,
        initializer_spec: Any,
    ) -> None:
        """Require components constructible independently for each trial."""
        for key in ("model", "model_type", "optimizer", "loss"):
            if key in model_config:
                cls._require_constructible(model_config[key], key)

        cls._require_constructible(
            constraint_spec,
            "constraint",
        )

        regularizer = parse_regularizer_spec(regularizer_spec)

        if regularizer is not None:
            cls._require_constructible(
                regularizer.selector,
                "regularizer",
            )

        cls._require_constructible(
            initializer_spec,
            "initializer",
        )

        if callbacks is not None:
            declarations = (
                callbacks
                if isinstance(callbacks, (list, tuple))
                else [callbacks]
            )

            for declaration in declarations:
                cls._require_constructible(
                    declaration,
                    "callback",
                )

    
    @staticmethod
    def _validate_score(
        result: ObjectiveResult,
        n_outputs: int,
    ) -> None:
        if not isinstance(result, ObjectiveResult):
            raise TypeError(
                "An objective must return ObjectiveResult."
            )

        if not np.isfinite(result.score):
            raise ValueError("Objective score must be finite.")

        if result.scores_by_output is not None:
            if (
                len(result.scores_by_output) != n_outputs
                or not all(
                    np.isfinite(value)
                    for value in result.scores_by_output
                )
            ):
                raise ValueError(
                    "Per-output scores must be finite and aligned."
                )

    @staticmethod
    def _validate_options(
        *,
        trial_id,
        validation_part,
        objective_scale,
        training_seed,
        error_policy,
    ) -> None:
        if (
            isinstance(trial_id, bool)
            or not isinstance(trial_id, int)
            or trial_id < 1
        ):
            raise ValueError("trial_id must be a positive integer.")

        if (
            not isinstance(validation_part, str)
            or not validation_part.strip()
            or validation_part != validation_part.strip()
            or validation_part in {"training", "test"}
        ):
            raise ValueError("Invalid validation part name.")

        if objective_scale not in {"scaled", "prepared_raw"}:
            raise ValueError("Unsupported objective scale.")

        if training_seed is not None and (
            isinstance(training_seed, bool)
            or not isinstance(training_seed, Integral)
            or training_seed < 0
        ):
            raise ValueError(
                "training_seed must be a non-negative integer or None."
            )

        if error_policy not in {"raise", "record"}:
            raise ValueError("Unsupported error policy.")

    def validate_inputs(
        self,
        *,
        prepared: PreparedData,
        objective: BaseObjective,
        validation_part: str = "validation",
        objective_scale: ObjectiveScale = "scaled",
    ) -> None:
        """Validate shared search inputs before constructing candidates."""
        if not isinstance(objective, BaseObjective):
            raise TypeError("objective must be a BaseObjective instance.")

        if objective.direction not in {"minimize", "maximize"}:
            raise ValueError("Unsupported objective direction.")

        if (
            not isinstance(validation_part, str)
            or not validation_part.strip()
            or validation_part != validation_part.strip()
            or validation_part in {"training", "test"}
        ):
            raise ValueError("Invalid validation part name.")

        if objective_scale not in {"scaled", "prepared_raw"}:
            raise ValueError("Unsupported objective scale.")

        _, validation = self._select_parts(
            prepared,
            validation_part,
        )

        if objective_scale == "prepared_raw":
            if (
                prepared.y_scaler is None
                or not callable(
                    getattr(prepared.y_scaler, "inverse_transform", None)
                )
            ):
                raise DataValidationError(
                    "prepared_raw scoring requires a fitted y_scaler "
                    "providing inverse_transform()."
                )

            if not np.isfinite(prepared.y_raw[validation]).all():
                raise DataValidationError(
                    "Raw validation targets must be finite."
                )