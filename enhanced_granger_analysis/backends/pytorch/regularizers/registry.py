from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components.regularizer_registry import RegularizerClassRegistry
from .base import BasePyTorchRegularizer


RegularizerClass = TypeVar(
    "RegularizerClass",
    bound=type[BasePyTorchRegularizer],
)


_REGULARIZERS = RegularizerClassRegistry[BasePyTorchRegularizer](
    base_class=BasePyTorchRegularizer,
    label="PyTorch regularizer",
)


def register_regularizer(
    *names: str,
) -> Callable[[RegularizerClass], RegularizerClass]:
    """Register a class; the first name is canonical."""
    return _REGULARIZERS.register(*names)


def get_regularizer_class(
    name: str,
) -> type[BasePyTorchRegularizer]:
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
    tuple[type[BasePyTorchRegularizer], dict[str, Any]]
    | BasePyTorchRegularizer
    | None
):
    """Resolve a raw or parsed specification; None disables regularization."""
    return _REGULARIZERS.resolve_spec(spec)