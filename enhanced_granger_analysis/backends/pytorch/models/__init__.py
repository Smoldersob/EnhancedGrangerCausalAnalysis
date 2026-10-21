from .registry import (
    register_model,
    get_model_class,
    available_models,
    resolve_model_spec,
)

# Import wykonuje dekorator rejestrujący model.
from .conditional import PyTorchGrangerModel


__all__ = [
    "PyTorchGrangerModel",
    "register_model",
    "get_model_class",
    "available_models",
    "resolve_model_spec",
]