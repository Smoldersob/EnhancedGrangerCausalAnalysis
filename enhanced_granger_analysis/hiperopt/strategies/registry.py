from __future__ import annotations

from collections.abc import Mapping
from inspect import isabstract

from ...core.registry import Registry
from .base import BaseSearchStrategy


_STRATEGIES: Registry[type[BaseSearchStrategy]] = Registry(
    label="search strategy",
)


def _validate_class(cls):
    if (
        not isinstance(cls, type)
        or not issubclass(cls, BaseSearchStrategy)
        or isabstract(cls)
    ):
        raise TypeError(
            "A strategy must be a concrete BaseSearchStrategy subclass."
        )


def register_search_strategy(*names):
    """Register a strategy class. The first name is canonical."""
    register = _STRATEGIES.decorator(*names)

    def decorator(cls):
        _validate_class(cls)
        return register(cls)

    return decorator


def get_search_strategy_class(name):
    """Resolve a name without constructing a strategy."""
    return _STRATEGIES.resolve(name)


def available_search_strategies(*, include_aliases=False):
    return _STRATEGIES.available(include_aliases=include_aliases)


def resolve_search_strategy(spec, **params):
    """Resolve a name, declaration, class or ready strategy."""
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown or "type" not in spec:
            raise ValueError(
                "A strategy declaration requires type and optional params."
            )

        selector = spec["type"]
        if not isinstance(selector, str) or not selector.strip():
            raise TypeError("strategy.type must be a registered name.")

        embedded = spec.get("params", {})
        if not isinstance(embedded, Mapping):
            raise TypeError("strategy.params must be a mapping.")

        if not all(isinstance(key, str) for key in embedded):
            raise TypeError("Strategy parameter names must be strings.")

        conflicts = set(embedded) & set(params)
        if conflicts:
            raise ValueError(
                f"Duplicate strategy parameters: {sorted(conflicts)!r}."
            )

        params = {**dict(embedded), **params}
        spec = selector

    if isinstance(spec, BaseSearchStrategy):
        if params:
            raise ValueError(
                "A ready strategy cannot receive constructor parameters."
            )
        return spec

    cls = (
        get_search_strategy_class(spec)
        if isinstance(spec, str)
        else spec
    )
    _validate_class(cls)
    return cls(**params)