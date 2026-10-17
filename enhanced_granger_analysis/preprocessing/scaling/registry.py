"""Scaler class registration and instance resolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ...core.protocols import Scaler
from ...core.registry import Registry


ScalerClass = TypeVar("ScalerClass", bound=type[Scaler])

_SCALERS: Registry[type[Scaler]] = Registry(label="scaler")

_REQUIRED_METHODS = (
    "fit_transform",
    "transform",
    "inverse_transform",
)


def _has_scaler_interface(value: Any) -> bool:
    """Check the presence of the required callable methods."""
    return all(
        callable(getattr(value, method, None))
        for method in _REQUIRED_METHODS
    )


def _validate_scaler_class(cls: Any) -> None:
    """Reject non-classes and classes missing the scaler interface."""
    if not isinstance(cls, type):
        raise TypeError(
            "Only scaler classes can be registered, not instances "
            "or factory functions."
        )

    if not _has_scaler_interface(cls):
        raise TypeError(
            f"Scaler class {cls.__name__!r} must provide callable "
            "fit_transform, transform and inverse_transform methods."
        )


def _validate_scaler_instance(value: Any) -> Scaler:
    """Validate a resolved scaler instance."""
    if isinstance(value, type) or not _has_scaler_interface(value):
        raise TypeError(
            "Expected a scaler instance with callable fit_transform, "
            "transform and inverse_transform methods."
        )

    return value


def register_scaler(
    *names: str,
) -> Callable[[ScalerClass], ScalerClass]:
    """Register a scaler class; the first name is canonical."""
    if not names:
        raise ValueError(
            "Scaler registration requires at least one name."
        )

    def decorator(cls: ScalerClass) -> ScalerClass:
        _validate_scaler_class(cls)

        _SCALERS.register(
            names[0],
            cls,
            aliases=names[1:],
        )
        return cls

    return decorator


def get_scaler_class(name: str) -> type[Scaler]:
    """Return a registered scaler class without constructing it."""
    return _SCALERS.resolve(name)


def available_scalers(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """List scaler names without constructing instances."""
    return _SCALERS.available(include_aliases=include_aliases)


def resolve_scaler(
    spec: Any = None,
    **params: Any,
) -> Scaler:
    """Resolve a scaler specification into an instance.

    Supported inputs:
        None:
            Construct the scaler registered as 'none'.
        Registered name:
            Construct the corresponding class with params.
        Scaler class:
            Construct it with params without requiring registration.
        Scaler instance:
            Return it unchanged; params must be empty.
        Factory:
            Invoke it with params and validate the returned instance.
            Factories are accepted directly, but cannot be registered.

    Constructor and factory exceptions propagate unchanged.
    """
    if spec is None:
        spec = "none"

    if isinstance(spec, str):
        cls = get_scaler_class(spec)
        return _validate_scaler_instance(cls(**params))

    if isinstance(spec, type):
        _validate_scaler_class(spec)
        return _validate_scaler_instance(spec(**params))

    if _has_scaler_interface(spec):
        if params:
            raise ValueError(
                "Constructor parameters cannot be supplied with an "
                "existing scaler instance. Pass a class or name instead."
            )
        return _validate_scaler_instance(spec)

    if callable(spec):
        return _validate_scaler_instance(spec(**params))

    raise TypeError(
        "Scaler must be None, a registered name, a scaler class, "
        "a scaler instance or a callable factory."
    )