from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ....core.components import ModelClassRegistry
from ....core.models.protocol import GrangerModelProtocol


ModelClass = TypeVar("ModelClass", bound=type)


_MODELS = ModelClassRegistry[GrangerModelProtocol](
    label="PyTorch model",
)



def register_model(
    *names: str,
) -> Callable[[ModelClass], ModelClass]:
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




def _require_torch_model(
    model: GrangerModelProtocol,
) -> GrangerModelProtocol:
    if model.backend != "pytorch" and model.backend != "torch":
        raise ValueError(
            f"Expected a PyTorch model, got backend {model.backend!r}."
        )

    return model


def validate_model_instance(
    model: Any,
) -> GrangerModelProtocol:
    validated = _MODELS.validate_instance(model)
    return _require_torch_model(validated)


def resolve_model_spec(
    spec: Any,
) -> tuple[
    type[GrangerModelProtocol] | GrangerModelProtocol | None,
    dict[str, Any],
]:
    target, params = _MODELS.resolve_spec(spec)

    if target is not None and not isinstance(target, type):
        _require_torch_model(target)

    return target, params