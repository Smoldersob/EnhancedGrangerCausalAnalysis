"""Lag-selector class registration and instance resolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence, Mapping
from typing import Any, Protocol, TypeVar

import pandas as pd

from ...core.registry import Registry
from ...core.lag_config import LagSpec

class LagSelector(Protocol):
    """Minimal interface required by lag-selector resolution."""

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outputs: Sequence[str] | None = None,
    ) -> LagSpec:
        """Select lags from one dataset."""
        ...


LagSelectorClass = TypeVar(
    "LagSelectorClass",
    bound=type[LagSelector],
)

_LAG_SELECTORS: Registry[type[LagSelector]] = Registry(
    label="lag selector",
)


def _has_lag_selector_interface(value: Any) -> bool:
    """Check whether the required selector methods are callable."""
    return callable(getattr(value, "fit", None))


def _validate_lag_selector_class(cls: Any) -> None:
    """Reject non-classes and classes missing the selector interface."""
    if not isinstance(cls, type):
        raise TypeError(
            "Only lag-selector classes can be registered, not instances "
            "or factory functions."
        )

    if not _has_lag_selector_interface(cls):
        raise TypeError(
            f"Lag-selector class {cls.__name__!r} must provide "
            "a callable fit method."
        )


def _validate_lag_selector_instance(value: Any) -> LagSelector:
    """Validate a resolved lag-selector instance."""
    if isinstance(value, type) or not _has_lag_selector_interface(value):
        raise TypeError(
            "Expected a lag-selector instance with a callable fit method."
        )

    return value


def register_lag_selector(
    *names: str,
) -> Callable[[LagSelectorClass], LagSelectorClass]:
    """Register a selector class; the first name is canonical."""
    if not names:
        raise ValueError(
            "Lag-selector registration requires at least one name."
        )

    def decorator(cls: LagSelectorClass) -> LagSelectorClass:
        _validate_lag_selector_class(cls)

        _LAG_SELECTORS.register(
            names[0],
            cls,
            aliases=names[1:],
        )
        return cls

    return decorator


def get_lag_selector_class(name: str) -> type[LagSelector]:
    """Return a registered selector class without constructing it."""
    return _LAG_SELECTORS.resolve(name)


def available_lag_selectors(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """List registered selector names without constructing instances."""
    return _LAG_SELECTORS.available(
        include_aliases=include_aliases,
    )


def resolve_lag_selector(
    spec: Any = None,
    **params: Any,
) -> LagSelector | None:
    """Resolve a lag-selector specification into an instance.

    Supported inputs:
        None:
            Disable automatic lag selection. Parameters must be empty.
        Registered name:
            Construct the corresponding class with params.
        Selector class:
            Construct it with params without requiring registration.
        Selector instance:
            Return it unchanged. Parameters must be empty.
        Factory:
            Invoke it with params and validate the returned instance.

    Only classes can be registered. Factories may be supplied directly.

    Constructor and factory exceptions propagate unchanged.
    """
    if spec is None:
        if params:
            raise ValueError(
                "Lag-selector parameters cannot be supplied when "
                "automatic lag selection is disabled."
            )
        return None

    if isinstance(spec, Mapping):
        unknown = set(spec).difference({"type", "params"})
        if unknown:
            raise ValueError(
                "Unknown lag-selector specification fields: "
                f"{sorted(unknown, key=repr)!r}."
            )

        if "type" not in spec:
            raise ValueError(
                "A lag-selector specification requires 'type'."
            )

        selector_name = spec["type"]
        if (
            not isinstance(selector_name, str)
            or not selector_name.strip()
        ):
            raise TypeError(
                "lag_selector.type must be a non-empty registered name."
            )

        embedded_params = spec.get("params", {})
        if not isinstance(embedded_params, Mapping):
            raise TypeError(
                "lag_selector.params must be a mapping."
            )

        if not all(
            isinstance(name, str)
            for name in embedded_params
        ):
            raise TypeError(
                "lag_selector.params keys must be strings."
            )

        duplicate_params = set(embedded_params).intersection(params)
        if duplicate_params:
            raise ValueError(
                "Lag-selector parameters were supplied both inside "
                "the specification and as separate parameters: "
                f"{sorted(duplicate_params)!r}."
            )

        params = {**embedded_params, **params}
        spec = selector_name

    if isinstance(spec, str):
        cls = get_lag_selector_class(spec)
        return _validate_lag_selector_instance(cls(**params))

    if isinstance(spec, type):
        _validate_lag_selector_class(spec)
        return _validate_lag_selector_instance(spec(**params))

    if _has_lag_selector_interface(spec):
        if params:
            raise ValueError(
                "Constructor parameters cannot be supplied with an "
                "existing lag-selector instance. Pass a class or "
                "registered name instead."
            )
        return _validate_lag_selector_instance(spec)

    if callable(spec):
        return _validate_lag_selector_instance(spec(**params))

    raise TypeError(
        "Lag selector must be None, a registered name, a selector "
        "class, a selector instance or a callable factory."
    )