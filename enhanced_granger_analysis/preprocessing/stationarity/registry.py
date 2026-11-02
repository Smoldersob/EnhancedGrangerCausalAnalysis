from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping, Callable
from typing import Any, TypeVar

from ...core.registry import Registry
from .base import BaseStationarityEnforcer

STATIONARITY_REGISTRY: Registry[
    type[BaseStationarityEnforcer]
] = Registry(label="stationarity enforcer")

EnforcerClass = TypeVar(
    "EnforcerClass",
    bound=type[BaseStationarityEnforcer],
)


def register_stationarity_enforcer(
    *names: str,
) -> Callable[[EnforcerClass], EnforcerClass]:
    """Register an enforcer class; the first name is canonical."""
    if not names:
        raise ValueError(
            "Stationarity registration requires at least one name."
        )

    def decorator(cls: EnforcerClass) -> EnforcerClass:
        if not inspect.isclass(cls) or not issubclass(
            cls,
            BaseStationarityEnforcer,
        ):
            raise TypeError(
                "Only BaseStationarityEnforcer subclasses "
                "can be registered."
            )

        if inspect.isabstract(cls):
            raise TypeError(
                "Cannot register an abstract stationarity enforcer."
            )

        STATIONARITY_REGISTRY.register(
            names[0],
            cls,
            aliases=names[1:],
        )
        return cls

    return decorator

def resolve_stationarity_enforcer(
    spec: Any,
    **params: Any,
) -> BaseStationarityEnforcer | None:
    """Resolve a specification into an independent enforcer instance."""

    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                f"Unknown stationarity specification keys: "
                f"{sorted(unknown, key=str)}."
            )

        if "type" not in spec:
            raise ValueError(
                "Stationarity specification requires 'type'."
            )

        embedded = spec.get("params", {})
        if not isinstance(embedded, Mapping):
            raise TypeError(
                "Stationarity specification 'params' must be a mapping."
            )

        if not all(isinstance(key, str) for key in embedded):
            raise TypeError(
                "Stationarity parameter names must be strings."
            )

        overlap = set(embedded) & set(params)
        if overlap:
            raise ValueError(
                "Stationarity parameters supplied twice: "
                f"{sorted(overlap)}."
            )

        selector = spec["type"]
        if not isinstance(selector, str):
            raise TypeError(
                "A mapping specification requires a registered "
                "string name in 'type'."
            )

        params = {**dict(embedded), **params}
        spec = selector

    if spec is None:
        if params:
            raise ValueError(
                "Stationarity parameters cannot be supplied "
                "when stationarity processing is disabled."
            )
        return None

    if isinstance(spec, str):
        cls = STATIONARITY_REGISTRY.resolve(spec)

    elif inspect.isclass(spec):
        if not issubclass(spec, BaseStationarityEnforcer):
            raise TypeError(
                "Stationarity class must inherit "
                "BaseStationarityEnforcer."
            )
        cls = spec

    elif isinstance(spec, BaseStationarityEnforcer):
        if params:
            raise ValueError(
                "Constructor parameters cannot be supplied "
                "with an existing enforcer instance."
            )
        return copy.deepcopy(spec)

    else:
        raise TypeError(
            "Stationarity selection must be None, a registered name, "
            "a {'type': ..., 'params': ...} specification, "
            "an enforcer class, or an enforcer instance."
        )

    if inspect.isabstract(cls):
        raise TypeError(
            "Cannot construct an abstract stationarity enforcer."
        )

    return cls(**params)