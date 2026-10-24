from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components.regularizer_registry import RegularizerClassRegistry
from .base import BaseNumpyRegularizer


RegularizerClass = TypeVar(
    "RegularizerClass",
    bound=type[BaseNumpyRegularizer],
)


_REGULARIZERS = RegularizerClassRegistry[BaseNumpyRegularizer](
    base_class=BaseNumpyRegularizer,
    label="Numpy regularizer",
)


def register_regularizer(
    *names: str,
) -> Callable[[RegularizerClass], RegularizerClass]:
    """Register a class; the first name is canonical."""
    return _REGULARIZERS.register(*names)


def get_regularizer_class(
    name: str,
) -> type[BaseNumpyRegularizer]:
    """Return a registered class without constructing it."""
    return _REGULARIZERS.get_class(name)


def available_regularizers(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    return _REGULARIZERS.available(
        include_aliases=include_aliases,
    )


def resolve_regularizer_spec(
    spec: Any,
) -> (
    tuple[type[BaseNumpyRegularizer], dict[str, Any]]
    | BaseNumpyRegularizer
    | None
):
    """Resolve a raw or parsed specification; None disables regularization."""
    return _REGULARIZERS.resolve_spec(spec)