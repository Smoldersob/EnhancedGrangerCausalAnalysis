"""Fit coordinator: resolve the authorized sample, then fit each component once."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from ..core.data.analysis_roles import VariableRoles
from ..core.data.prepared import PreparedData
from ..core.exceptions import DataValidationError
from .configuration.plan import FittingPlan
from .fitted import FittedPreprocessor
from .splitting.scope import FitScope, FitSegment
from .splitting import resolve_data_splitter
from .sampling import sample_prepared_data


def _frames(
    data: pd.DataFrame | Sequence[pd.DataFrame],
) -> list[pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        result = [data.copy()]
    elif isinstance(data, Sequence) and all(isinstance(frame, pd.DataFrame) for frame in data):
        result = [frame.copy() for frame in data]
    else:
        raise DataValidationError("Fit data must be a DataFrame or a sequence of DataFrames.")
    if not result:
        raise DataValidationError("Fit data must not be empty.")
    return result


def _validate_numeric_strategy(plan: FittingPlan, frames: Sequence[pd.DataFrame]) -> None:
    selected = set(plan.spec.requested_roles.causes) | set(plan.spec.requested_roles.effects)
    for frame_index, frame in enumerate(frames):
        for name in selected:
            if name not in frame.columns:
                continue
            series = frame[name]
            if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
                raise DataValidationError(
                    f"Variable {name!r} uses the numeric-continuous strategy but "
                    f"DataFrame {frame_index} has dtype {series.dtype!s}."
                )


def _scope_from_source_splitter(
    frames: Sequence[pd.DataFrame],
    splitter_spec: object,
) -> FitScope:
    splitter = resolve_data_splitter(splitter_spec)
    if splitter is None or splitter.stage != "source":
        raise DataValidationError(
            "A prepared-stage splitter cannot define a fitting scope; "
            "supply an explicit source FitScope."
        )
    splitter.fit(frames)
    segments = tuple(
        FitSegment(
            dataset_index=block.dataset_index,
            row_start=block.row_start,
            row_stop=block.row_stop,
            labels=tuple(frames[block.dataset_index].index[block.row_start:block.row_stop]),
        )
        for block in splitter.fitting_blocks
    )
    return FitScope(segments=segments)


def fit_preprocessor(
    plan: FittingPlan,
    frames: pd.DataFrame | Sequence[pd.DataFrame],
    *,
    scope: FitScope | None = None,
) -> tuple[FittedPreprocessor, PreparedData]:
    """Return fitted transform state and its separate prepared training result."""
    if not isinstance(plan, FittingPlan):
        raise TypeError("plan must be a FittingPlan.")
    inputs = _frames(frames)
    _validate_numeric_strategy(plan, inputs)

    active_scope = scope
    if active_scope is None:
        splitter = resolve_data_splitter(plan.spec.config.splitter)
        if splitter is not None and splitter.stage == "source":
            active_scope = _scope_from_source_splitter(
                inputs,
                plan.spec.config.splitter,
            )
        elif splitter is not None:
            raise DataValidationError(
                "A prepared-stage splitter cannot authorize fitting decisions. "
                "Pass an explicit FitScope based on independent source rows."
            )
        else:
            active_scope = FitScope.all_data(inputs)

    active_scope = active_scope.bind(inputs)
    active_scope.select_frames(inputs)

    # Import lazily to keep planning independent of execution components.
    from .pipeline import _PreparationExecutor

    executor = _PreparationExecutor(config=plan.spec.config)
    executor.fit(
        inputs,
        roles=plan.spec.requested_roles,
        scope=active_scope,
        variable_descriptions=plan.spec.variable_descriptions,
        apply_sampling=False,
    )
    training_data = executor.prepared_data_.as_prepared_data()
    fitted = FittedPreprocessor.from_executor(
        executor,
        plan,
        active_scope,
        training_data,
    )
    training_data = sample_prepared_data(
        training_data,
        fraction=plan.spec.config.backend_sample_fraction,
        max_samples=plan.spec.config.backend_max_samples,
        random_state=plan.spec.config.random_state,
        strategy=plan.spec.config.sampling_strategy,
    )
    return fitted, training_data


__all__ = ["fit_preprocessor"]
