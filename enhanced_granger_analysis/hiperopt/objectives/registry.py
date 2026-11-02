"""Objective registration and specification resolution."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import isabstract
from typing import Any

from ...core.registry import Registry
from .base import BaseObjective


_OBJECTIVES: Registry[type[BaseObjective]] = Registry(
    label="objective",
)


def _validate_objective_class(cls: Any) -> None:
    """Validate an objective class without constructing it."""
    if (
        not isinstance(cls, type)
        or not issubclass(cls, BaseObjective)
        or isabstract(cls)
    ):
        raise TypeError(
            "An objective must be a concrete BaseObjective subclass."
        )


def register_objective(*names: str):
    """Register a class. The first name is canonical."""
    register = _OBJECTIVES.decorator(*names)

    def decorator(cls):
        _validate_objective_class(cls)
        return register(cls)

    return decorator


def get_objective_class(name: str) -> type[BaseObjective]:
    """Return a registered class without constructing it."""
    return _OBJECTIVES.resolve(name)


def available_objectives(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """Return registered objective names."""
    return _OBJECTIVES.available(include_aliases=include_aliases)


def resolve_objective(
    spec: Any,
    **params: Any,
) -> BaseObjective:
    """Resolve a declaration, class or ready objective instance."""
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                "Unknown objective specification fields: "
                f"{sorted(unknown, key=repr)!r}."
            )

        if "type" not in spec:
            raise ValueError(
                "An objective specification requires 'type'."
            )

        selector = spec["type"]
        if not isinstance(selector, str) or not selector.strip():
            raise TypeError(
                "objective.type must be a non-empty registered name."
            )

        embedded = spec.get("params", {})
        if not isinstance(embedded, Mapping):
            raise TypeError("objective.params must be a mapping.")

        if not all(isinstance(key, str) for key in embedded):
            raise TypeError(
                "Objective parameter names must be strings."
            )

        conflicts = set(embedded) & set(params)
        if conflicts:
            raise ValueError(
                "Objective parameters were supplied twice: "
                f"{sorted(conflicts)!r}."
            )

        params = {**dict(embedded), **params}
        spec = selector

    if isinstance(spec, BaseObjective):
        if params:
            raise ValueError(
                "A ready objective cannot receive constructor parameters."
            )
        objective = spec
    else:
        cls = (
            get_objective_class(spec)
            if isinstance(spec, str)
            else spec
        )
        _validate_objective_class(cls)
        objective = cls(**params)

    if objective.direction not in {"minimize", "maximize"}:
        raise ValueError(
            "Objective direction must be 'minimize' or 'maximize'."
        )

    return objective