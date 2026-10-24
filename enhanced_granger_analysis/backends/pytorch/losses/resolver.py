from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_ALIASES = {
    "mse": "MSELoss",
    "mean_squared_error": "MSELoss",
    "mae": "L1Loss",
    "l1": "L1Loss",
    "mean_absolute_error": "L1Loss",
    "huber": "HuberLoss",
    "smooth_l1": "SmoothL1Loss",
    "bce": "BCELoss",
    "binary_crossentropy": "BCELoss",
    "bce_with_logits": "BCEWithLogitsLoss",
    "binary_crossentropy_with_logits": "BCEWithLogitsLoss",
    "cross_entropy": "CrossEntropyLoss",
    "nll": "NLLLoss",
    "poisson_nll": "PoissonNLLLoss",
    "gaussian_nll": "GaussianNLLLoss",
    "kl_div": "KLDivLoss",
}


def _normalize_name(name: str) -> str:
    return name.strip().replace("_", "").replace("-", "").lower()


def _get_loss_class(name: str) -> type:
    import torch.nn as nn

    if not name.strip():
        raise ValueError("Loss name must not be empty.")

    class_name = _ALIASES.get(name.strip().lower(), name.strip())
    normalized = _normalize_name(class_name)

    for exported_name in dir(nn):
        if (
            exported_name.startswith("_")
            or not exported_name.endswith("Loss")
            or _normalize_name(exported_name) != normalized
        ):
            continue

        candidate = getattr(nn, exported_name)

        if (
            isinstance(candidate, type)
            and issubclass(candidate, nn.Module)
        ):
            return candidate

    raise ValueError(
        f"Unknown PyTorch loss class {name!r}. "
        "Use a built-in loss name or pass a custom module "
        "class, instance or function directly."
    )


def resolve_loss(spec: Any = None) -> Any:
    """Return a ready PyTorch loss or None.

    Accepted:
        None;
        built-in name;
        {"type": name, "params": {...}};
        nn.Module instance;
        concrete nn.Module class;
        loss function or callable object.

    Callable factories are not invoked.
    """
    if spec is None:
        return None

    import torch.nn as nn

    if isinstance(spec, nn.Module):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, nn.Module) or spec is nn.Module:
            raise TypeError(
                "Loss class must be a concrete subclass "
                "of torch.nn.Module."
            )

        return spec()

    if isinstance(spec, str):
        return _get_loss_class(spec)()

    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                f"Unknown loss specification fields: {unknown!r}."
            )

        name = spec.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "loss.type must be a non-empty string."
            )

        params = spec.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError("loss.params must be a mapping.")

        if any(not isinstance(key, str) for key in params):
            raise TypeError(
                "Loss parameter names must be strings."
            )

        return _get_loss_class(name)(**dict(params))

    if callable(spec):
        return spec

    raise TypeError(
        "Loss must be None, a name, a specification mapping, "
        "a torch.nn.Module instance, a module class or a function."
    )