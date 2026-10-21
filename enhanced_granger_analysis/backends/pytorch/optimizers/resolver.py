from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from inspect import isabstract
from typing import Any, TypeAlias

import torch
from torch.optim import Optimizer


ParameterInput: TypeAlias = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
)

OptimizerFactory: TypeAlias = Callable[[ParameterInput], Optimizer]
ResolvedOptimizer: TypeAlias = Optimizer | OptimizerFactory


def _validate_optimizer_class(value: Any) -> type[Optimizer]:
    if not isinstance(value, type) or not issubclass(value, Optimizer):
        raise TypeError(
            "Optimizer class must inherit from torch.optim.Optimizer."
        )

    if isabstract(value) or value is Optimizer:
        raise TypeError("A concrete optimizer class is required.")

    return value


def _get_optimizer_class(name: str) -> type[Optimizer]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Optimizer name must be a non-empty string.")

    available = {
        key.lower(): value
        for key, value in vars(torch.optim).items()
        if (
            isinstance(value, type)
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


def _construct_optimizer(
    parameters: ParameterInput,
    *,
    provider: Callable[..., Optimizer],
    options: Mapping[str, Any],
) -> Optimizer:
    optimizer = provider(parameters, **dict(options))

    if not isinstance(optimizer, Optimizer):
        raise TypeError(
            "Optimizer factory must return torch.optim.Optimizer."
        )

    return optimizer


def resolve_optimizer(
    spec: Any,
    *,
    default_lr: float | None = None,
) -> ResolvedOptimizer:
    """Return an existing optimizer or a factory accepting model parameters.

    Supported specs:
        None:
            Adam with native defaults, optionally default_lr.
        str:
            Name exported by torch.optim.
        class:
            Concrete Optimizer subclass.
        mapping:
            {"type": name_or_class, "params": {...}}
        callable:
            Factory called as factory(parameters, **options).
        instance:
            Returned unchanged; its parameter binding must be checked later.
    """
    if isinstance(spec, Optimizer):
        if default_lr is not None:
            raise ValueError(
                "Do not combine an optimizer instance with default_lr. "
                "Configure the instance explicitly."
            )
        return spec

    options: dict[str, Any] = {}

    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                f"Unknown optimizer specification keys: {unknown!r}."
            )

        if "type" not in spec:
            raise ValueError("Optimizer specification requires 'type'.")

        raw_options = spec.get("params", {})
        if not isinstance(raw_options, Mapping):
            raise TypeError("Optimizer 'params' must be a mapping.")

        options = dict(raw_options)
        spec = spec["type"]

    if any(not isinstance(key, str) for key in options):
        raise TypeError("Optimizer parameter names must be strings.")

    if "learning_rate" in options:
        if "lr" in options:
            raise ValueError("Use either 'lr' or 'learning_rate', not both.")
        options["lr"] = options.pop("learning_rate")

    forbidden = {
        "params",
        "closure",
        "gradient_accumulation_steps",
        "gradient_accumulation",
        "accumulation_steps",
    } & set(options)

    if forbidden:
        raise ValueError(
            "These are not optimizer constructor options: "
            + ", ".join(sorted(forbidden))
        )

    if default_lr is not None:
        options.setdefault("lr", default_lr)

    if spec is None:
        provider = torch.optim.Adam
    elif isinstance(spec, str):
        provider = _get_optimizer_class(spec)
    elif isinstance(spec, type):
        provider = _validate_optimizer_class(spec)
    elif callable(spec):
        provider = spec
    else:
        raise TypeError(
            "Optimizer must be None, a name, configuration mapping, "
            "Optimizer subclass, factory or instance."
        )

    return partial(
        _construct_optimizer,
        provider=provider,
        options=options,
    )


def validate_optimizer_binding(
    optimizer: Optimizer,
    parameters: Iterable[torch.Tensor],
) -> None:
    """Check that an optimizer refers to exactly the intended parameters."""
    expected = list(parameters)
    actual = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    expected_ids = [id(parameter) for parameter in expected]
    actual_ids = [id(parameter) for parameter in actual]

    if (
        len(expected_ids) != len(actual_ids)
        or set(expected_ids) != set(actual_ids)
    ):
        raise ValueError(
            "Optimizer is bound to different parameters. "
            "Pass a class/configuration/factory for a newly created network."
        )