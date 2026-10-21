from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components.callback_registry import CallbackClassRegistry
from .base import Callback


CallbackClass = TypeVar("CallbackClass", bound=type[Callback])


_CALLBACKS = CallbackClassRegistry[Callback](
    base_class=Callback,
    label="Scikit-learn callback",
)


def register_callback(
    *names: str,
) -> Callable[[CallbackClass], CallbackClass]:
    return _CALLBACKS.register(*names)


def get_callback_class(name: str) -> type[Callback]:
    return _CALLBACKS.get_class(name)


def available_callbacks(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    return _CALLBACKS.available(include_aliases=include_aliases)


def resolve_callback_spec(
    spec: Any,
) -> tuple[type[Callback], dict[str, Any]] | Callback | None:
    return _CALLBACKS.resolve_spec(spec)