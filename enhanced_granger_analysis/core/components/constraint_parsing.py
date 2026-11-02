from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from ..exceptions import ConstraintConfigurationError
from .constraint_spec import (
    MinAbsSumRelationRule,
    RelationConstraint,
    RelationConstraintsSpec,
    ZeroRelationRule,
)


RelationsInput: TypeAlias = (
    RelationConstraintsSpec | list[Mapping[str, Any]]
)


def _check_fields(
    value: Mapping,
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    non_string_keys = [
        key for key in value
        if not isinstance(key, str)
    ]
    if non_string_keys:
        raise ConstraintConfigurationError(
            f"{context}: all field names must be strings"
        )

    present = set(value)
    missing = required - present
    unknown = present - required - optional

    if missing:
        raise ConstraintConfigurationError(
            f"{context}: missing fields {sorted(missing)}"
        )

    if unknown:
        raise ConstraintConfigurationError(
            f"{context}: unknown fields {sorted(unknown)}"
        )


def _parse_rule(
    raw: Any,
    *,
    context: str,
) -> ZeroRelationRule | MinAbsSumRelationRule:
    if not isinstance(raw, Mapping):
        raise ConstraintConfigurationError(
            f"{context} must be an object"
        )

    _check_fields(
        raw,
        required={"type"},
        optional={"params"},
        context=context,
    )

    rule_type = raw["type"]
    if not isinstance(rule_type, str):
        raise ConstraintConfigurationError(
            f"{context}.type must be a string"
        )

    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise ConstraintConfigurationError(
            f"{context}.params must be an object"
        )

    if rule_type == "zero":
        _check_fields(
            params,
            required=set(),
            optional=set(),
            context=f"{context}.params",
        )
        return ZeroRelationRule()

    if rule_type == "min_abs_sum":
        _check_fields(
            params,
            required={"value"},
            optional=set(),
            context=f"{context}.params",
        )
        return MinAbsSumRelationRule(
            value=params["value"],
        )

    raise ConstraintConfigurationError(
        f"{context}.type: unsupported value {rule_type!r}; "
        "expected 'zero' or 'min_abs_sum'"
    )


def parse_relations_spec(
    raw: RelationsInput,
) -> RelationConstraintsSpec:
    """Parse a ready specification or a list of canonical declarations."""
    if isinstance(raw, RelationConstraintsSpec):
        return raw

    if not isinstance(raw, list):
        raise ConstraintConfigurationError(
            "relations must be a RelationConstraintsSpec instance "
            "or a list of relation declaration objects"
        )

    relations = []

    for index, declaration in enumerate(raw):
        context = f"relations[{index}]"

        if not isinstance(declaration, Mapping):
            raise ConstraintConfigurationError(
                f"{context} must be an object"
            )

        _check_fields(
            declaration,
            required={"cause", "effect", "rule"},
            optional=set(),
            context=context,
        )

        rule = _parse_rule(
            declaration["rule"],
            context=f"{context}.rule",
        )

        relations.append(RelationConstraint(
            cause=declaration["cause"],
            effect=declaration["effect"],
            rule=rule,
        ))

    return RelationConstraintsSpec(
        relations=tuple(relations),
    )