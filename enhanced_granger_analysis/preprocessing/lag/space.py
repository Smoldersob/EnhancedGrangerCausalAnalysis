"""Resolved finite search spaces for lag selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import TypeAlias

from ...core.lags.specification import LagSpec

PairLagMap: TypeAlias = Mapping[str, Mapping[str, tuple[int, ...]]]
PairReasonMap: TypeAlias = Mapping[
    str, Mapping[str, Mapping[int, tuple[str, ...]]]
]


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be a sequence of names, not a string.")
    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must contain non-empty names.")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate names.")
    return names


def _lag_tuple(values: Sequence[int], context: str) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{context} must be an explicit tuple or list.")
    if any(
        isinstance(lag, bool) or not isinstance(lag, Integral) or lag < 0
        for lag in values
    ):
        raise ValueError(f"{context} must contain non-negative integers.")
    result = tuple(int(lag) for lag in values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{context} must be sorted with no duplicates.")
    return result


def _validate_pair_map(
    values: PairLagMap | None,
    *,
    outputs: tuple[str, ...],
    predictors: tuple[str, ...],
    label: str,
) -> dict[str, dict[str, tuple[int, ...]]]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping indexed by output.")
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for output, row in values.items():
        if output not in outputs:
            raise ValueError(f"{label} references unknown output {output!r}.")
        if not isinstance(row, Mapping):
            raise TypeError(f"{label}[{output!r}] must be a mapping.")
        result[output] = {}
        for predictor, lags in row.items():
            if predictor not in predictors:
                raise ValueError(
                    f"{label}[{output!r}] references unknown predictor "
                    f"{predictor!r}."
                )
            result[output][predictor] = _lag_tuple(
                lags, f"{label}[{output!r}][{predictor!r}]"
            )
    return result


@dataclass(frozen=True)
class LagSelectionSpace:
    """Finite, resolved candidate lags for every output/predictor pair.

    ``allowed_lags`` is complete: an empty tuple means no candidates, while
    unrestricted input constraints are expanded to the finite configured
    lag domain by :meth:`resolve`. ``fixed_lags`` is partial; a present pair
    is an exact relation, including an explicitly fixed empty relation.
    """

    outputs: tuple[str, ...]
    predictors: tuple[str, ...]
    allowed_lags: PairLagMap
    required_lags: PairLagMap
    fixed_lags: PairLagMap
    exclusion_reasons: PairReasonMap
    constraint_origins: Mapping[str, Mapping[str, tuple[str, ...]]]

    def __post_init__(self) -> None:
        outputs = _names(self.outputs, "outputs")
        predictors = _names(self.predictors, "predictors")
        allowed = _validate_pair_map(
            self.allowed_lags,
            outputs=outputs,
            predictors=predictors,
            label="allowed_lags",
        )
        required = _validate_pair_map(
            self.required_lags,
            outputs=outputs,
            predictors=predictors,
            label="required_lags",
        )
        fixed = _validate_pair_map(
            self.fixed_lags,
            outputs=outputs,
            predictors=predictors,
            label="fixed_lags",
        )
        for label, values in (
            ("allowed_lags", allowed),
            ("required_lags", required),
        ):
            for output in outputs:
                missing = set(predictors).difference(values.get(output, {}))
                extra = set(values.get(output, {})).difference(predictors)
                if missing or extra:
                    raise ValueError(
                        f"{label} must explicitly include every pair; "
                        f"invalid row for {output!r}."
                    )

        for output in outputs:
            for predictor in predictors:
                permitted = set(allowed[output][predictor])
                required_pair = set(required[output][predictor])
                if not required_pair.issubset(permitted):
                    raise ValueError(
                        f"Required lags for {output!r} <- {predictor!r} "
                        "must be allowed."
                    )
                if output == predictor and 0 in permitted:
                    raise ValueError(
                        f"Own-variable lag 0 is not allowed for {output!r}."
                    )
                if output == predictor and 0 in required_pair:
                    raise ValueError(
                        f"Own-variable lag 0 cannot be required for {output!r}."
                    )
                if predictor in fixed.get(output, {}):
                    exact = set(fixed[output][predictor])
                    if not exact.issubset(permitted):
                        raise ValueError(
                            f"Fixed lags for {output!r} <- {predictor!r} "
                            "must be allowed."
                        )
                    if not required_pair.issubset(exact):
                        raise ValueError(
                            f"Fixed lags for {output!r} <- {predictor!r} "
                            "must include all required lags."
                        )

        reasons: dict[str, Mapping[str, Mapping[int, tuple[str, ...]]]] = {}
        if not isinstance(self.exclusion_reasons, Mapping):
            raise TypeError("exclusion_reasons must be a mapping.")
        for output, row in self.exclusion_reasons.items():
            if output not in outputs or not isinstance(row, Mapping):
                raise ValueError("exclusion_reasons contains an invalid output.")
            reasons[output] = MappingProxyType({})
            normalized_row: dict[str, Mapping[int, tuple[str, ...]]] = {}
            for predictor, by_lag in row.items():
                if predictor not in predictors or not isinstance(by_lag, Mapping):
                    raise ValueError("exclusion_reasons contains an invalid pair.")
                normalized_lags: dict[int, tuple[str, ...]] = {}
                for lag, reason in by_lag.items():
                    if isinstance(lag, bool) or not isinstance(lag, Integral) or lag < 0:
                        raise ValueError("Exclusion-reason keys must be non-negative lags.")
                    reason_tuple = (reason,) if isinstance(reason, str) else tuple(reason)
                    if not reason_tuple or any(not isinstance(item, str) or not item for item in reason_tuple):
                        raise ValueError("Exclusion reasons must be non-empty strings.")
                    if lag in allowed[output][predictor]:
                        raise ValueError(
                            f"Excluded lag {lag} is also allowed for "
                            f"{output!r} <- {predictor!r}."
                        )
                    normalized_lags[int(lag)] = reason_tuple
                normalized_row[predictor] = MappingProxyType(normalized_lags)
            reasons[output] = MappingProxyType(normalized_row)

        if not isinstance(self.constraint_origins, Mapping):
            raise TypeError("constraint_origins must be a mapping.")
        origins: dict[str, Mapping[str, tuple[str, ...]]] = {}
        for output, row in self.constraint_origins.items():
            if output not in outputs or not isinstance(row, Mapping):
                raise ValueError("constraint_origins contains an invalid output.")
            normalized_row: dict[str, tuple[str, ...]] = {}
            for predictor, sources in row.items():
                if predictor not in predictors:
                    raise ValueError("constraint_origins contains an invalid pair.")
                source_tuple = (sources,) if isinstance(sources, str) else tuple(sources)
                if any(not isinstance(source, str) or not source for source in source_tuple):
                    raise ValueError("Constraint origins must be non-empty strings.")
                normalized_row[predictor] = source_tuple
            origins[output] = MappingProxyType(normalized_row)

        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "predictors", predictors)
        object.__setattr__(self, "allowed_lags", MappingProxyType({
            output: MappingProxyType(allowed[output]) for output in outputs
        }))
        object.__setattr__(self, "required_lags", MappingProxyType({
            output: MappingProxyType(required[output]) for output in outputs
        }))
        object.__setattr__(self, "fixed_lags", MappingProxyType({
            output: MappingProxyType(fixed.get(output, {})) for output in outputs
        }))
        object.__setattr__(self, "exclusion_reasons", MappingProxyType(reasons))
        object.__setattr__(self, "constraint_origins", MappingProxyType(origins))

    @classmethod
    def resolve(
        cls,
        *,
        outputs: Sequence[str],
        predictors: Sequence[str],
        max_lag: int,
        use_lag_zero: bool = False,
        allowed_lags: PairLagMap | None = None,
        required_lags: PairLagMap | None = None,
        fixed_lags: PairLagMap | None = None,
        exclusion_reasons: PairReasonMap | None = None,
        constraint_origins: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    ) -> "LagSelectionSpace":
        """Resolve partial restrictions into a complete finite lag space."""
        outputs = _names(outputs, "outputs")
        predictors = _names(predictors, "predictors")
        if isinstance(max_lag, bool) or not isinstance(max_lag, Integral) or max_lag < 0:
            raise ValueError("max_lag must be a non-negative integer.")
        if not isinstance(use_lag_zero, bool):
            raise TypeError("use_lag_zero must be a bool.")

        raw_allowed = _validate_pair_map(
            allowed_lags, outputs=outputs, predictors=predictors, label="allowed_lags"
        )
        raw_required = _validate_pair_map(
            required_lags, outputs=outputs, predictors=predictors, label="required_lags"
        )
        raw_fixed = _validate_pair_map(
            fixed_lags, outputs=outputs, predictors=predictors, label="fixed_lags"
        )
        domain = tuple(
            lag for lag in range(0 if use_lag_zero else 1, int(max_lag) + 1)
        )
        resolved_allowed: dict[str, dict[str, tuple[int, ...]]] = {}
        resolved_required: dict[str, dict[str, tuple[int, ...]]] = {}
        resolved_fixed: dict[str, dict[str, tuple[int, ...]]] = {}
        resolved_reasons = {
            output: {
                predictor: dict(
                    (exclusion_reasons or {}).get(output, {}).get(predictor, {})
                )
                for predictor in predictors
            }
            for output in outputs
        }

        for output in outputs:
            resolved_allowed[output] = {}
            resolved_required[output] = {}
            resolved_fixed[output] = {}
            for predictor in predictors:
                pair_domain = tuple(
                    lag for lag in domain
                    if not (output == predictor and lag == 0)
                )
                explicit = raw_allowed.get(output, {}).get(predictor)
                if explicit is None:
                    candidates = pair_domain
                else:
                    outside = set(explicit).difference(pair_domain)
                    if outside:
                        raise ValueError(
                            f"Allowed lags outside the configured domain for "
                            f"{output!r} <- {predictor!r}: {sorted(outside)!r}."
                        )
                    candidates = explicit
                    for lag in set(pair_domain).difference(candidates):
                        resolved_reasons[output][predictor].setdefault(
                            lag, ("excluded by explicit allowed_lags",)
                        )
                required = raw_required.get(output, {}).get(predictor, ())
                if not set(required).issubset(candidates):
                    raise ValueError(
                        f"Required lags for {output!r} <- {predictor!r} "
                        "must be in allowed_lags."
                    )
                resolved_allowed[output][predictor] = candidates
                resolved_required[output][predictor] = required
                if predictor in raw_fixed.get(output, {}):
                    exact = raw_fixed[output][predictor]
                    if not set(exact).issubset(candidates):
                        raise ValueError(
                            f"Fixed lags for {output!r} <- {predictor!r} "
                            "must be allowed."
                        )
                    if not set(required).issubset(exact):
                        raise ValueError(
                            f"Fixed lags for {output!r} <- {predictor!r} "
                            "must include all required lags."
                        )
                    resolved_fixed[output][predictor] = exact

        return cls(
            outputs=outputs,
            predictors=predictors,
            allowed_lags=resolved_allowed,
            required_lags=resolved_required,
            fixed_lags=resolved_fixed,
            exclusion_reasons=resolved_reasons,
            constraint_origins=constraint_origins or {},
        )

    def for_predictors(self, predictors: Sequence[str]) -> "LagSelectionSpace":
        """Project onto new predictor axes, closing newly-added pairs.

        Selection APIs that require response columns among their data columns
        use this to make such response-only candidate pairs explicitly empty.
        """
        expanded = _names(predictors, "predictors")
        if not set(self.predictors).issubset(expanded):
            raise ValueError("Expanded predictors must retain the existing axes.")
        allowed = {
            output: {
                predictor: self.allowed_lags[output].get(predictor, ())
                for predictor in expanded
            }
            for output in self.outputs
        }
        required = {
            output: {
                predictor: self.required_lags[output].get(predictor, ())
                for predictor in expanded
            }
            for output in self.outputs
        }
        fixed = {
            output: {
                predictor: lags
                for predictor, lags in self.fixed_lags[output].items()
            }
            for output in self.outputs
        }
        reasons = {
            output: {
                predictor: dict(self.exclusion_reasons.get(output, {}).get(predictor, {}))
                for predictor in expanded
                if predictor in self.exclusion_reasons.get(output, {})
            }
            for output in self.outputs
        }
        origins = {
            output: {
                predictor: values
                for predictor, values in self.constraint_origins.get(output, {}).items()
            }
            for output in self.outputs
        }
        return LagSelectionSpace(
            self.outputs, expanded, allowed, required, fixed, reasons, origins
        )

    def validate_selection(self, specification: LagSpec, *, context: str) -> None:
        """Reject selected relations that violate this hard search space."""
        for output in self.outputs:
            for predictor in self.predictors:
                selected = set(specification[output][predictor])
                allowed = set(self.allowed_lags[output][predictor])
                required = set(self.required_lags[output][predictor])
                if not selected.issubset(allowed):
                    raise ValueError(
                        f"{context} selects forbidden lags for "
                        f"{output!r} <- {predictor!r}: "
                        f"{sorted(selected.difference(allowed))!r}."
                    )
                if not required.issubset(selected):
                    raise ValueError(
                        f"{context} omits required lags for "
                        f"{output!r} <- {predictor!r}: "
                        f"{sorted(required.difference(selected))!r}."
                    )
                if predictor in self.fixed_lags[output]:
                    fixed = set(self.fixed_lags[output][predictor])
                    if selected != fixed:
                        raise ValueError(
                            f"{context} must preserve fixed relation "
                            f"{output!r} <- {predictor!r} exactly."
                        )


__all__ = ["LagSelectionSpace"]
