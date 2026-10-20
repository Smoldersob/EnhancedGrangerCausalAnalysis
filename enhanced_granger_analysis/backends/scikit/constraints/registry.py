from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components.constraint_registry import ConstraintClassRegistry
from .base import ScikitConstraint


ConstraintClass = TypeVar(
    "ConstraintClass",
    bound=type[ScikitConstraint],
)

_CONSTRAINTS = ConstraintClassRegistry[ScikitConstraint](
    base_class=ScikitConstraint,
    label="NumPy constraint",
)


def register_constraint(
    *names: str,
) -> Callable[[ConstraintClass], ConstraintClass]:
    return _CONSTRAINTS.register(*names)


def get_constraint_class(name: str) -> type[ScikitConstraint]:
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
) -> tuple[type[ScikitConstraint], dict[str, Any]]:
    return _CONSTRAINTS.resolve_spec(spec)