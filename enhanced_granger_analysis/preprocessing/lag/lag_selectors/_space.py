"""Validation helpers for the resolved finite lag-selection space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..space import LagSelectionSpace


class LagSelectionCapabilityError(ValueError):
    """A selector cannot represent the supplied lag-selection space."""


@dataclass(frozen=True)
class ResolvedLagSpace:
    outputs: tuple[str, ...]
    predictors: tuple[str, ...]
    optional: dict[str, dict[str, tuple[int, ...]]]
    required: dict[str, dict[str, tuple[int, ...]]]
    fixed: dict[str, dict[str, tuple[int, ...]]]

    def mandatory(self, output: str, predictor: str) -> tuple[int, ...]:
        return tuple(sorted(set(
            self.required[output][predictor]
            + self.fixed[output].get(predictor, ())
        )))


def resolve_lag_space(
    space: LagSelectionSpace | None,
    outputs: Sequence[str] | None,
    data_columns: Sequence[str],
) -> ResolvedLagSpace | None:
    """Check that the plan axes match this call and its observed columns."""
    if space is None:
        return None
    if not isinstance(space, LagSelectionSpace):
        raise TypeError("space must be a LagSelectionSpace or None.")
    if outputs is None:
        raise ValueError("outputs must be provided when space is supplied.")

    requested_outputs = tuple(outputs)
    if requested_outputs != space.outputs:
        raise ValueError(
            "outputs must exactly match space.outputs in order."
        )

    names = set(data_columns)
    missing = (set(space.outputs) | set(space.predictors)).difference(names)
    if missing:
        raise ValueError(
            f"LagSelectionSpace references columns absent from data: "
            f"{sorted(missing)!r}."
        )

    return ResolvedLagSpace(
        outputs=space.outputs,
        predictors=space.predictors,
        optional={
            output: {
                predictor: (
                    ()
                    if predictor in space.fixed_lags[output]
                    else tuple(
                        lag
                        for lag in space.allowed_lags[output][predictor]
                        if lag not in set(
                            space.required_lags[output][predictor]
                        )
                    )
                )
                for predictor in space.predictors
            }
            for output in space.outputs
        },
        required={
            output: dict(space.required_lags[output])
            for output in space.outputs
        },
        fixed={
            output: dict(space.fixed_lags[output])
            for output in space.outputs
        },
    )


def validate_method_lags(
    resolved: ResolvedLagSpace,
    *,
    use_lag_zero: bool,
    max_lag: int,
) -> None:
    """Reject plans that request lags outside the strategy's candidate range."""
    for output in resolved.outputs:
        for predictor in resolved.predictors:
            optional = resolved.optional[output][predictor]
            mandatory = resolved.mandatory(output, predictor)
            if output == predictor and 0 in (*optional, *mandatory):
                raise LagSelectionCapabilityError(
                    f"Selector cannot use own lag 0 for {output!r}."
                )
            if not use_lag_zero and 0 in (*optional, *mandatory):
                raise LagSelectionCapabilityError(
                    "LagSelectionSpace includes cross-variable lag 0, "
                    "but this selector has use_lag_zero=False."
                )
            outside = tuple(lag for lag in optional if lag > max_lag)
            if outside:
                raise LagSelectionCapabilityError(
                    f"Optional lags {outside!r} for {output!r} <- "
                    f"{predictor!r} exceed selector max_lag={max_lag}."
                )


__all__ = [
    "LagSelectionCapabilityError",
    "ResolvedLagSpace",
    "resolve_lag_space",
    "validate_method_lags",
]
