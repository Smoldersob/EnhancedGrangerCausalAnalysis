"""Coordinate candidate generation, evaluation and model selection."""

from __future__ import annotations

from itertools import islice
from time import perf_counter
from types import MappingProxyType

from ..configuration.modeling_spec import ModelingSpec
from ..configuration.model_search_spec import ModelSearchSpec
from ..hiperopt.objectives import resolve_objective
from ..hiperopt.results import (
    ModelSearchFailedError,
    ModelSearchResult,
    TrialResult,
)
from ..hiperopt.space import SearchSpaceParser
from ..hiperopt.space.base import SearchSpace
from ..hiperopt.strategies import resolve_search_strategy
from ..modeling.search_configuration import (
    apply_modeling_candidate,
)
from ..modeling.search_evaluator import ModelCandidateEvaluator
from ..preprocessing.prepare_data import PreparedData


class ModelSearchRunner:
    """Run a bounded search without fitting Granger reference models."""

    def __init__(
        self,
        *,
        evaluator: ModelCandidateEvaluator | None = None,
        space_parser: SearchSpaceParser | None = None,
    ) -> None:
        self._evaluator = (
            evaluator
            if evaluator is not None
            else ModelCandidateEvaluator()
        )
        self._space_parser = (
            space_parser
            if space_parser is not None
            else SearchSpaceParser()
        )

    def run(
        self,
        *,
        spec: ModelSearchSpec,
        modeling: ModelingSpec,
        prepared: PreparedData,
        preparation: Mapping[str, object] | None = None,
    ) -> ModelSearchResult:
        """Evaluate candidates and return selected modeling settings."""
        started = perf_counter()

        space = (
            spec.space
            if isinstance(spec.space, SearchSpace)
            else self._space_parser.parse(spec.space)
        )

        # Validate paths even for a directly supplied SearchSpace.
        self._validate_space_paths(space)

        objective = resolve_objective(spec.objective)
        strategy = resolve_search_strategy(spec.strategy)

        if spec.budget.epochs_per_trial is not None:
            self._validate_budget_paths(space)

        if prepared.split is None:
            raise ValueError(
                "Model search requires a prepared data split."
            )

        split_snapshot = MappingProxyType({
            name: tuple(int(index) for index in indices)
            for name, indices in prepared.split.parts.items()
        })

        trials: list[TrialResult] = []
        best: TrialResult | None = None

        candidates = islice(
            strategy.candidates(space),
            int(spec.budget.max_trials),
        )

        for trial_id, candidate in enumerate(candidates, start=1):
            trial = self._evaluator.evaluate(
                trial_id=trial_id,
                candidate=candidate,
                modeling=modeling,
                prepared=prepared,
                objective=objective,
                validation_part=spec.validation_part,
                objective_scale=spec.objective_scale,
                training_seed=spec.training_seed,
                error_policy=spec.error_policy,
                collect_history=spec.collect_history,
                epochs_per_trial=spec.budget.epochs_per_trial,
            )

            trials.append(trial)

            if trial.status != "completed":
                continue

            if best is None or self._is_better(
                trial,
                best,
                objective.direction,
            ):
                best = trial

        if best is None:
            raise ModelSearchFailedError(
                "Model search produced no successful candidate.",
                trials=tuple(trials),
            )

        best_modeling = apply_modeling_candidate(
            modeling,
            best.candidate,
        )

        return ModelSearchResult(
            trials=tuple(trials),
            best_trial_id=best.trial_id,
            best_modeling=best_modeling,
            direction=objective.direction,
            objective_scale=spec.objective_scale,
            validation_part=spec.validation_part,
            budget=spec.budget,
            split_parts=split_snapshot,
            stop_reason=(
                "budget_reached"
                if len(trials) == spec.budget.max_trials
                else "space_exhausted"
            ),
            duration_seconds=perf_counter() - started,
            preparation=preparation,
        )

    @staticmethod
    def _is_better(
        candidate: TrialResult,
        current: TrialResult,
        direction: str,
    ) -> bool:
        assert candidate.objective is not None
        assert current.objective is not None

        if direction == "minimize":
            return candidate.objective.score < current.objective.score

        return candidate.objective.score > current.objective.score

    def _validate_space_paths(self, space: SearchSpace) -> None:
        declarations = {}

        for parameter in space.parameters:
            path = parameter.path

            if (
                not isinstance(path, tuple)
                or len(path) < 2
                or any(
                    not isinstance(part, str)
                    or not part
                    or "." in part
                    for part in path
                )
            ):
                raise ValueError(
                    f"Invalid search parameter path: {path!r}."
                )

            name = ".".join(path)

            if name in declarations:
                raise ValueError(
                    f"Repeated search parameter path: {name!r}."
                )

            declarations[name] = {"choices": [None]}

        self._space_parser.parse(declarations)

    @staticmethod
    def _validate_budget_paths(space: SearchSpace) -> None:
        """Reject search dimensions conflicting with a fixed epoch limit."""
        protected = (
            ("model_config", "epochs"),
            ("model_config", "max_iter"),
            ("model_config", "model", "params", "epochs"),
            ("model_config", "model", "params", "max_iter"),
            ("model_config", "model_type", "params", "epochs"),
            ("model_config", "model_type", "params", "max_iter"),
        )

        for parameter in space.parameters:
            for path in protected:
                common = min(len(parameter.path), len(path))

                if parameter.path[:common] == path[:common]:
                    raise ValueError(
                        "Search dimension conflicts with the fixed "
                        "trial epoch budget: "
                        f"{'.'.join(parameter.path)!r}."
                    )