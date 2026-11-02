from __future__ import annotations

from collections.abc import Mapping
from inspect import isabstract
from typing import Any

import torch.optim as optim
from torch.optim import Optimizer

from .adapter import PyTorchOptimizerAdapter


def _validate_optimizer_class(
    value: Any,
) -> type[Optimizer]:
    if (
        not isinstance(value, type)
        or not issubclass(value, Optimizer)
        or value is Optimizer
        or isabstract(value)
    ):
        raise TypeError(
            "Optimizer class must be a concrete "
            "torch.optim.Optimizer subclass."
        )

    return value


def _get_optimizer_class(
    name: str,
) -> type[Optimizer]:
    if not name.strip():
        raise ValueError(
            "Optimizer name must be a non-empty string."
        )

    available = {
        key.lower(): value
        for key, value in vars(optim).items()
        if (
            not key.startswith("_")
            and isinstance(value, type)
            and issubclass(value, Optimizer)
            and value is not Optimizer
            and not isabstract(value)
        )
    }

    try:
        return available[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown PyTorch optimizer {name!r}. "
            f"Available: {', '.join(sorted(available))}."
        ) from exc


def resolve_optimizer(
    spec: Any = None,
) -> PyTorchOptimizerAdapter:
    """Return a new, unbound optimizer adapter.

    Supported:
        None -> Adam with native defaults.
        str -> optimizer exported by torch.optim.
        class -> concrete Optimizer subclass.
        mapping -> {"type": name_or_class, "params": {...}}.

    Native optimizer instances, adapter instances and callable factories
    are intentionally unsupported.
    """
    options: dict[str, Any] = {}

    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                f"Unknown optimizer specification keys: {unknown!r}."
            )

        if "type" not in spec:
            raise ValueError(
                "Optimizer specification requires 'type'."
            )

        raw_options = spec.get("params", {})
        if not isinstance(raw_options, Mapping):
            raise TypeError(
                "Optimizer 'params' must be a mapping."
            )

        options = dict(raw_options)
        selector = spec["type"]

        if selector is None:
            raise ValueError(
                "Optimizer 'type' cannot be None. "
                "Pass None as the whole specification "
                "to select the default optimizer."
            )
    else:
        selector = spec

    if any(not isinstance(key, str) for key in options):
        raise TypeError(
            "Optimizer parameter names must be strings."
        )

    forbidden = {
        "params",
        "closure",
        "gradient_accumulation_steps",
        "gradient_accumulation",
        "accumulation_steps",
    } & set(options)

    if forbidden:
        raise ValueError(
            "These are not supported optimizer constructor options: "
            + ", ".join(sorted(forbidden))
        )

    if isinstance(selector, Optimizer):
        raise TypeError(
            "Native optimizer instances are not supported. "
            "Pass a name, class or configuration."
        )

    if isinstance(selector, PyTorchOptimizerAdapter):
        raise TypeError(
            "Pass the optimizer specification, not an adapter instance."
        )

    if selector is None:
        optimizer_class = optim.Adam
    elif isinstance(selector, str):
        optimizer_class = _get_optimizer_class(selector)
    elif isinstance(selector, type):
        optimizer_class = _validate_optimizer_class(selector)
    else:
        raise TypeError(
            "Optimizer must be None, a registered native name, "
            "an Optimizer subclass or a configuration mapping. "
            "Callable factories are not supported."
        )

    return PyTorchOptimizerAdapter(
        optimizer_class,
        options=options,
    )