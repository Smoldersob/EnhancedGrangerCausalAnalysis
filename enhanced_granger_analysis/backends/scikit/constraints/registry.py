from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components.constraint_registry import ConstraintClassRegistry
from .base import BaseScikitConstraint


ConstraintClass = TypeVar(
    "ConstraintClass",
    bound=type[BaseScikitConstraint],
)

_CONSTRAINTS = ConstraintClassRegistry[BaseScikitConstraint](
    base_class=BaseScikitConstraint,
    label="Scikit-learn constraint",
)


def register_constraint(
    *names: str,
) -> Callable[[ConstraintClass], ConstraintClass]:
    return _CONSTRAINTS.register(*names)


def get_constraint_class(name: str) -> type[BaseScikitConstraint]:
    return _CONSTRAINTS.get_class(name)


def available_constraints(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    return _CONSTRAINTS.available(
        include_aliases=include_aliases,
    )


def resolve_constraint_spec(
    spec: Any,
) -> (
    tuple[type[BaseScikitConstraint], dict[str, Any]]
    | BaseScikitConstraint
    | None
):
    return _CONSTRAINTS.resolve_spec(spec)