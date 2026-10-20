from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import isabstract
from typing import Any, Generic, TypeVar

from ..exceptions import RegularizerConfigurationError
from ..registry import Registry
from .regularizer_parsing import parse_regularizer_spec
from .regularizer_spec import RegularizerSpec

RegularizerBase = TypeVar("RegularizerBase")
RegularizerClass = TypeVar("RegularizerClass", bound=type)


class RegularizerClassRegistry(Generic[RegularizerBase]):
    """Backend-specific class registry and specification resolver.

    This object never constructs regularizer instances.
    """

    def __init__(
        self,
        *,
        base_class: type[RegularizerBase],
        label: str,
    ) -> None:
        self._base_class = base_class
        self._registry: Registry[type[RegularizerBase]] = Registry(
            label=label,
        )

    def _validate_class(
        self,
        cls: Any,
    ) -> type[RegularizerBase]:
        if not isinstance(cls, type):
            raise TypeError(
                "Only regularizer classes are accepted, "
                "not instances or factory functions."
            )

        if not issubclass(cls, self._base_class):
            raise TypeError(
                f"Regularizer class {cls.__name__!r} must inherit from "
                f"{self._base_class.__name__}."
            )

        if isabstract(cls):
            raise TypeError(
                f"Regularizer class {cls.__name__!r} is abstract "
                "and cannot be registered or selected."
            )

        return cls

    def register(
        self,
        *names: str,
    ) -> Callable[[RegularizerClass], RegularizerClass]:
        """Register a class; the first name is canonical."""
        if not names:
            raise ValueError(
                "Regularizer registration requires at least one name."
            )

        def decorator(
            cls: RegularizerClass,
        ) -> RegularizerClass:
            validated = self._validate_class(cls)

            self._registry.register(
                names[0],
                validated,
                aliases=names[1:],
            )

            return cls

        return decorator

    def get_class(
        self,
        name: str,
    ) -> type[RegularizerBase]:
        """Return a registered class without constructing it."""
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
    ) -> tuple[type[RegularizerBase], dict[str, Any]] | RegularizerBase | None:
        """Resolve an optional specification without constructing an instance.

        Raw specifications are parsed by the shared parser.
        Prepared RegularizerSpec objects are used directly.
        """
        if spec is None:
            return None

        if isinstance(spec, self._base_class):
            return spec

        normalized = (
            spec
            if isinstance(spec, RegularizerSpec)
            else parse_regularizer_spec(spec)
        )

        if normalized is None:
            return None

        selector = normalized.selector

        if isinstance(selector, str):
            cls = self.get_class(selector)
        else:
            cls = self._validate_class(selector)

        return cls, dict(normalized.params)