"""Initializer class registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, Mapping

from ..core.protocols import Initializer
from ..core.registry import Registry


InitializerClass = TypeVar(
    "InitializerClass",
    bound=type[Initializer],
)

_INITIALIZERS: Registry[type[Initializer]] = Registry(
    label="initializer"
)


def _validate_initializer_class(cls: Any) -> None:
    if not isinstance(cls, type):
        raise TypeError(
            "Only initializer classes can be registered, "
            "not instances or factory functions."
        )

    if not callable(getattr(cls, "__call__", None)):
        raise TypeError(
            f"Initializer class {cls.__name__!r} "
            "must implement __call__()."
        )


def register_initializer(
    *names: str,
) -> Callable[[InitializerClass], InitializerClass]:
    """Register a class; the first name is canonical."""
    if not names:
        raise ValueError(
            "Initializer registration requires at least one name."
        )

    def decorator(cls: InitializerClass) -> InitializerClass:
        _validate_initializer_class(cls)

        _INITIALIZERS.register(
            names[0],
            cls,
            aliases=names[1:],
        )
        return cls

    return decorator


def get_initializer_class(name: str) -> type[Initializer]:
    """Return a registered class without constructing it."""
    return _INITIALIZERS.resolve(name)


def available_initializers(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """List registered names without constructing instances."""
    return _INITIALIZERS.available(
        include_aliases=include_aliases
    )


_SPEC_KEYS = frozenset({"type", "params"})

_RESERVED_PARAMS = frozenset({
    "n_targets",
    "n_features_eff",
    "use_lag_zero",
    "lag0_indices",
    "Y",
    "X_lagged",
    "mask",
})


def resolve_initializer_spec(
    spec: Any,
) -> tuple[type[Initializer] | None, dict[str, Any]]:
    """Resolve an initializer specification without constructing it.

    Supported inputs:
        None:
            No additional weight initialization.
        str:
            Registered initializer name or alias.
        Mapping:
            {"type": registered_name, "params": {...}}
        class:
            Initializer class; registration is not required.

    Instances and factory functions are not accepted.
    """
    if spec is None:
        return None, {}

    if isinstance(spec, str):
        return get_initializer_class(spec), {}

    if isinstance(spec, Mapping):
        unknown_keys = set(spec) - _SPEC_KEYS
        if unknown_keys:
            raise ValueError(
                "Unknown initializer specification keys: "
                + ", ".join(sorted(map(str, unknown_keys)))
            )

        name = spec.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "initializer.type must be a non-empty string."
            )

        raw_params = spec.get("params", {})
        if raw_params is None:
            raw_params = {}

        if not isinstance(raw_params, Mapping):
            raise TypeError(
                "initializer.params must be a mapping."
            )

        params = dict(raw_params)

        if not all(isinstance(key, str) for key in params):
            raise TypeError(
                "initializer.params keys must be strings."
            )

        reserved = set(params) & _RESERVED_PARAMS
        if reserved:
            raise ValueError(
                "Initializer parameters reserved for execution context: "
                + ", ".join(sorted(reserved))
            )

        return get_initializer_class(name), params

    if isinstance(spec, type):
        _validate_initializer_class(spec)
        return spec, {}

    raise TypeError(
        "Initializer specification must be None, a registered name, "
        "a mapping with 'type' and optional 'params', "
        "or an initializer class. "
        "Instances and factory functions are not supported."
    )