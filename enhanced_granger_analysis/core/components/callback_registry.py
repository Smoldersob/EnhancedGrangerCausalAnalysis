from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import isabstract
from typing import Any, Generic, TypeVar

from ..registry import Registry


CallbackBase = TypeVar("CallbackBase")
CallbackClass = TypeVar("CallbackClass", bound=type)


class CallbackClassRegistry(Generic[CallbackBase]):
    """Class registry and spec resolver; never constructs callbacks."""

    def __init__(
        self,
        *,
        base_class: type[CallbackBase],
        label: str,
    ) -> None:
        self._base_class = base_class
        self._registry: Registry[type[CallbackBase]] = Registry(
            label=label,
        )

    def _validate_class(self, cls: Any) -> type[CallbackBase]:
        if not isinstance(cls, type):
            raise TypeError("Only callback classes may be registered.")

        if not issubclass(cls, self._base_class):
            raise TypeError(
                f"Callback class must inherit from "
                f"{self._base_class.__name__}."
            )

        if isabstract(cls):
            raise TypeError("An abstract callback cannot be selected.")

        return cls

    def register(
        self,
        *names: str,
    ) -> Callable[[CallbackClass], CallbackClass]:
        if not names:
            raise ValueError("At least one callback name is required.")

        def decorator(cls: CallbackClass) -> CallbackClass:
            validated = self._validate_class(cls)
            self._registry.register(
                names[0],
                validated,
                aliases=names[1:],
            )
            return cls

        return decorator

    def get_class(self, name: str) -> type[CallbackBase]:
        return self._registry.resolve(name)

    def available(
        self,
        *,
        include_aliases: bool = False,
    ) -> tuple[str, ...]:
        return self._registry.available(
            include_aliases=include_aliases,
        )

    def resolve_spec(
        self,
        spec: Any,
    ) -> tuple[type[CallbackBase], dict[str, Any]] | CallbackBase | None:
        if spec is None:
            return None

        if isinstance(spec, self._base_class):
            return spec

        if isinstance(spec, str):
            return self.get_class(spec), {}

        if isinstance(spec, type):
            return self._validate_class(spec), {}

        if isinstance(spec, Mapping):
            if any(not isinstance(key, str) for key in spec):
                raise ValueError("Callback specification keys must be strings.")

            unknown = set(spec) - {"type", "params"}
            if unknown:
                raise ValueError(
                    "Unknown callback specification keys: "
                    + ", ".join(sorted(unknown))
                )

            name = spec.get("type")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Callback specification requires a non-empty 'type'."
                )

            params = spec.get("params", {})
            if not isinstance(params, Mapping):
                raise TypeError("Callback 'params' must be a mapping.")

            if any(not isinstance(key, str) for key in params):
                raise TypeError("Callback parameter names must be strings.")

            return self.get_class(name), dict(params)

        raise TypeError(
            "Callback specification must be None, a name, mapping, "
            "callback class or callback instance."
        )