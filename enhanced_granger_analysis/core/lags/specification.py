from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np

from ..data.feature_layout import validate_names


LagSpec: TypeAlias = dict[str, dict[str, tuple[int, ...]]]
"""Selected lag tuples indexed [output][predictor], including explicit empty pairs."""
LagFeature: TypeAlias = tuple[str, int]
Aggregation: TypeAlias = Literal["union", "intersection", "frequency"]


def _lags(values) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise TypeError("Lag values must be an explicit sequence of integers.")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError("Lag values must be one-dimensional.")
    normalized = []
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError("Lags must be non-negative integers.")
        normalized.append(int(value))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate lags are not allowed.")
    return tuple(sorted(normalized))


def _pairs(spec, *, allow_self_zero: bool = False) -> LagSpec:
    if not isinstance(spec, Mapping):
        raise TypeError("Lag entries must be a mapping indexed by output and predictor.")
    result: LagSpec = {}
    for output, by_predictor in spec.items():
        validate_names((output,), "Output")
        if not isinstance(by_predictor, Mapping):
            raise TypeError(f"Lag entries for output {output!r} must be a mapping.")
        row = {}
        for predictor, values in by_predictor.items():
            validate_names((predictor,), "Predictor")
            try:
                row[predictor] = _lags(values)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{output!r} <- {predictor!r}: {exc}") from exc
            if output == predictor and not allow_self_zero and 0 in row[predictor]:
                raise ValueError(f"Self lag 0 is not allowed for {output!r}.")
        result[output] = row
    return result


def validate_lag_spec(spec, *, outputs, predictors) -> LagSpec:
    """Validate a complete selection and return a canonical ordered copy."""
    outputs = validate_names(outputs, "Outputs")
    predictors = validate_names(predictors, "Predictors")
    values = _pairs(spec)
    if set(values) != set(outputs):
        raise ValueError("LagSpec must contain exactly the specified outputs.")
    for output in outputs:
        if set(values[output]) != set(predictors):
            raise ValueError(f"LagSpec for {output!r} must contain exactly the specified predictors.")
    return {
        output: {predictor: values[output][predictor] for predictor in predictors}
        for output in outputs
    }


@dataclass(frozen=True)
class StructuralLagConstraints:
    """Partial pre-selection restrictions; neither overrides nor selected lags.

    Missing allowed entry inherits ``default_allowed_lags`` (``None`` means
    unrestricted at this layer). An explicit empty allowed tuple closes a pair.
    Missing required/forbidden entries impose nothing beyond global forbiddance.
    """

    allowed_lags: Mapping[str, Mapping[str, tuple[int, ...]]] = field(default_factory=dict)
    forbidden_lags: Mapping[str, Mapping[str, tuple[int, ...]]] = field(default_factory=dict)
    required_lags: Mapping[str, Mapping[str, tuple[int, ...]]] = field(default_factory=dict)
    default_allowed_lags: tuple[int, ...] | None = None
    global_forbidden_lags: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        default = self.default_allowed_lags
        default = None if default is None else _lags(default)
        global_forbidden = _lags(self.global_forbidden_lags)
        allowed = _pairs(self.allowed_lags)
        forbidden = _pairs(self.forbidden_lags, allow_self_zero=True)
        required = _pairs(self.required_lags)
        for output, by_predictor in required.items():
            for predictor, lags in by_predictor.items():
                permitted = allowed.get(output, {}).get(predictor, default)
                denied = set(global_forbidden).union(forbidden.get(output, {}).get(predictor, ()))
                for lag in lags:
                    if lag in denied:
                        raise ValueError(f"{output!r} <- {predictor!r}: required lag {lag} is forbidden.")
                    if permitted is not None and lag not in permitted:
                        raise ValueError(f"{output!r} <- {predictor!r}: required lag {lag} is not allowed.")
        for name, pairs in (
            ("allowed_lags", allowed), ("forbidden_lags", forbidden), ("required_lags", required),
        ):
            object.__setattr__(self, name, MappingProxyType({
                output: MappingProxyType(row) for output, row in pairs.items()
            }))
        object.__setattr__(self, "default_allowed_lags", default)
        object.__setattr__(self, "global_forbidden_lags", global_forbidden)
