from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components.model_registry import ModelClassRegistry
from ....core.models.protocol import GrangerModelProtocol


ModelClass = TypeVar("ModelClass", bound=type)


_MODELS = ModelClassRegistry[GrangerModelProtocol](
    label="scikit-learn model",
)


def register_model(
    *names: str,
) -> Callable[[ModelClass], ModelClass]:
    """Register a class; the first name is canonical."""
    return _MODELS.register(*names)


def get_model_class(
    name: str,
) -> type[GrangerModelProtocol]:
    return _MODELS.get_class(name)


def available_models(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    return _MODELS.available(
        include_aliases=include_aliases,
    )


def resolve_model_spec(
    spec: Any,
) -> (
    tuple[type[GrangerModelProtocol], dict[str, Any]]
    | GrangerModelProtocol
    | None
):
    return _MODELS.resolve_spec(spec)