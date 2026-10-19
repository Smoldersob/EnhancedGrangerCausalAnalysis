from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Literal

import numpy as np

from ..exceptions import ConstraintConfigurationError
from .constraint_spec import (
    MinAbsSumRelationRule,
    PreparedConstraintData,
    PreparedMinAbsSumRule,
    RelationConstraintsSpec,
    ZeroRelationRule,
)


def _validate_layout(
    cause_slices: Mapping[str, slice],
    n_features: int,
) -> dict[str, slice]:
    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, Integral)
        or n_features < 0
    ):
        raise ConstraintConfigurationError(
            "n_features must be a non-negative integer"
        )

    coverage = np.zeros(n_features, dtype=np.int64)
    result = {}

    for name, block in cause_slices.items():
        if not isinstance(name, str) or not name.strip():
            raise ConstraintConfigurationError(
                "cause_slices keys must be non-empty strings"
            )

        if not isinstance(block, slice):
            raise ConstraintConfigurationError(
                f"Expected slice for cause {name!r}"
            )

        start, stop = block.start, block.stop

        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, Integral)
            or not isinstance(stop, Integral)
            or block.step not in (None, 1)
            or not 0 <= start <= stop <= n_features
        ):
            raise ConstraintConfigurationError(
                f"Invalid slice for cause {name!r}: {block!r}"
            )

        normalized = slice(int(start), int(stop))
        result[name] = normalized
        coverage[normalized] += 1

    if np.any(coverage != 1):
        raise ConstraintConfigurationError(
            "cause_slices must cover every feature exactly once"
        )

    return result


def prepare_constraint_data(
    spec: RelationConstraintsSpec,
    *,
    cause_slices: Mapping[str, slice],
    output_names: Sequence[str],
    n_features: int,
    base_mask: np.ndarray | None = None,
    infeasible_policy: Literal["error", "warn"] = "error",
) -> PreparedConstraintData:
    """Compile named relations into backend-independent indexed data."""
    if not isinstance(spec, RelationConstraintsSpec):
        raise ConstraintConfigurationError(
            "spec must be RelationConstraintsSpec; "
            "call parse_relations_spec() first"
        )

    if infeasible_policy not in {"error", "warn"}:
        raise ConstraintConfigurationError(
            "infeasible_policy must be 'error' or 'warn'"
        )

    if isinstance(output_names, (str, bytes)):
        raise ConstraintConfigurationError(
            "output_names must be a sequence of names"
        )

    outputs = tuple(output_names)

    if (
        not outputs
        or any(
            not isinstance(name, str) or not name.strip()
            for name in outputs
        )
        or len(set(outputs)) != len(outputs)
    ):
        raise ConstraintConfigurationError(
            "output_names must contain unique, non-empty strings"
        )

    blocks = _validate_layout(cause_slices, n_features)
    n_features = int(n_features)
    expected_shape = (len(outputs), n_features)

    if base_mask is None:
        mask = np.ones(expected_shape, dtype=bool)
    else:
        raw_mask = np.asarray(base_mask)

        if raw_mask.shape != expected_shape:
            raise ConstraintConfigurationError(
                f"base_mask shape {raw_mask.shape} "
                f"does not match {expected_shape}"
            )

        if raw_mask.dtype.kind not in "biuf":
            raise ConstraintConfigurationError(
                "base_mask must be boolean or numeric"
            )

        if not np.all((raw_mask == 0) | (raw_mask == 1)):
            raise ConstraintConfigurationError(
                "base_mask must contain only 0 and 1"
            )

        mask = raw_mask.astype(bool, copy=True)

    output_indices = {name: i for i, name in enumerate(outputs)}

    for relation in spec.relations:
        if relation.cause not in blocks:
            raise ConstraintConfigurationError(
                f"Unknown effective cause: {relation.cause!r}"
            )
        if relation.effect not in output_indices:
            raise ConstraintConfigurationError(
                f"Unknown effective effect: {relation.effect!r}"
            )

    # Apply all hard-zero requirements before generating indexed rules.
    for relation in spec.relations:
        if isinstance(relation.rule, ZeroRelationRule):
            row = output_indices[relation.effect]
            mask[row, blocks[relation.cause]] = False

    rules = []

    for relation in spec.relations:
        if not isinstance(relation.rule, MinAbsSumRelationRule):
            continue

        # A zero lower bound adds no restriction.
        if relation.rule.value == 0:
            continue

        row = output_indices[relation.effect]
        block = blocks[relation.cause]
        indices = np.arange(block.start, block.stop, dtype=np.int64)
        active = indices[mask[row, indices]]

        if active.size == 0:
            message = (
                f"Cannot enforce min_abs_sum={relation.rule.value} "
                f"for cause={relation.cause!r}, "
                f"effect={relation.effect!r}: "
                "no active coefficients remain"
            )

            if infeasible_policy == "error":
                raise ConstraintConfigurationError(message)

            warnings.warn(message, UserWarning, stacklevel=2)
            continue

        rules.append(PreparedMinAbsSumRule(
            output_index=row,
            feature_indices=tuple(int(index) for index in active),
            min_abs_sum=relation.rule.value,
        ))

    return PreparedConstraintData(
        output_names=outputs,
        n_features=n_features,
        mask=mask,
        rules=tuple(rules),
    )