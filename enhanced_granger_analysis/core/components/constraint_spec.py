from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ConstraintConfigurationError


def _validate_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstraintConfigurationError(
            f"{field} must be a non-empty string"
        )
    # Preserve the exact variable name, including meaningful whitespace.
    return value


@dataclass(frozen=True)
class ZeroRelationRule:
    type: Literal["zero"] = "zero"

    def __post_init__(self) -> None:
        if self.type != "zero":
            raise ConstraintConfigurationError("Invalid zero rule type")


@dataclass(frozen=True)
class MinAbsSumRelationRule:
    value: float
    type: Literal["min_abs_sum"] = "min_abs_sum"

    def __post_init__(self) -> None:
        if self.type != "min_abs_sum":
            raise ConstraintConfigurationError("Invalid min_abs_sum rule type")

        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise ConstraintConfigurationError(
                "min_abs_sum.value must be a number, not bool"
            )

        value = float(self.value)
        if not isfinite(value) or value < 0:
            raise ConstraintConfigurationError(
                "min_abs_sum.value must be finite and >= 0"
            )

        object.__setattr__(self, "value", value)


RelationRule = ZeroRelationRule | MinAbsSumRelationRule


@dataclass(frozen=True)
class RelationConstraint:
    cause: str
    effect: str
    rule: RelationRule

    def __post_init__(self) -> None:
        _validate_name(self.cause, field="cause")
        _validate_name(self.effect, field="effect")

        if not isinstance(
            self.rule,
            (ZeroRelationRule, MinAbsSumRelationRule),
        ):
            raise ConstraintConfigurationError("Unsupported relation rule")


@dataclass(frozen=True)
class RelationConstraintsSpec:
    relations: tuple[RelationConstraint, ...] = ()

    def __post_init__(self) -> None:
        relations = tuple(self.relations)
        seen: set[tuple[str, str]] = set()

        for relation in relations:
            if not isinstance(relation, RelationConstraint):
                raise ConstraintConfigurationError(
                    "relations must contain RelationConstraint objects"
                )

            key = (relation.cause, relation.effect)
            if key in seen:
                raise ConstraintConfigurationError(
                    f"Duplicate relation: cause={key[0]!r}, effect={key[1]!r}"
                )
            seen.add(key)

        object.__setattr__(self, "relations", relations)

    def to_config(self) -> list[dict]:
        declarations = []

        for relation in self.relations:
            if isinstance(relation.rule, ZeroRelationRule):
                rule = {"type": "zero"}
            else:
                rule = {
                    "type": "min_abs_sum",
                    "params": {
                        "value": relation.rule.value,
                    },
                }

            declarations.append({
                "cause": relation.cause,
                "effect": relation.effect,
                "rule": rule,
            })

        return declarations


@dataclass(frozen=True)
class PreparedMinAbsSumRule:
    output_index: int
    feature_indices: tuple[int, ...]
    min_abs_sum: float


@dataclass(frozen=True, eq=False)
class PreparedConstraintData:
    """Backend-independent constraint data.

    mask uses the canonical shape (n_outputs, n_features).
    Objects should be created through prepare_constraint_data().
    """

    output_names: tuple[str, ...]
    n_features: int
    mask: NDArray[np.bool_]
    rules: tuple[PreparedMinAbsSumRule, ...]

    def __post_init__(self) -> None:
        output_names = tuple(self.output_names)
        mask = np.array(self.mask, dtype=bool, copy=True)

        expected = (len(output_names), self.n_features)
        if mask.shape != expected:
            raise ConstraintConfigurationError(
                f"mask shape {mask.shape} does not match {expected}"
            )

        mask.setflags(write=False)

        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "rules", tuple(self.rules))