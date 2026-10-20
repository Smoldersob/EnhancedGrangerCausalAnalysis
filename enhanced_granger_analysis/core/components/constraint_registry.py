from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import isabstract
from typing import Any, Generic, TypeVar

from ..exceptions import ConstraintConfigurationError
from ..registry import Registry


ConstraintBase = TypeVar("ConstraintBase")
ConstraintClass = TypeVar("ConstraintClass", bound=type)


_SPEC_KEYS = frozenset({"type", "params"})

_RESERVED_PARAMS = frozenset({
    "data",
    "mask",
    "rules",
    "output_names",
    "n_features",
    "cause_slices",
})


class ConstraintClassRegistry(Generic[ConstraintBase]):
    """Backend-specific class registry and specification resolver.

    This object never constructs constraint instances.
    """

    def __init__(
        self,
        *,
        base_class: type[ConstraintBase],
        label: str,
    ) -> None:
        self._base_class = base_class
        self._registry: Registry[type[ConstraintBase]] = Registry(
            label=label,
        )

    def _validate_class(
        self,
        cls: Any,
    ) -> type[ConstraintBase]:
        if not isinstance(cls, type):
            raise TypeError(
                "Only constraint classes are accepted, "
                "not instances or factory functions."
            )

        if not issubclass(cls, self._base_class):
            raise TypeError(
                f"Constraint class {cls.__name__!r} must inherit from "
                f"{self._base_class.__name__}."
            )

        if isabstract(cls):
            raise TypeError(
                f"Constraint class {cls.__name__!r} is abstract "
                "and cannot be registered or selected."
            )

        return cls

    def register(
        self,
        *names: str,
    ) -> Callable[[ConstraintClass], ConstraintClass]:
        """Register a class; the first name is canonical."""
        if not names:
            raise ValueError(
                "Constraint registration requires at least one name."
            )

        def decorator(cls: ConstraintClass) -> ConstraintClass:
            validated = self._validate_class(cls)

            self._registry.register(
                names[0],
                validated,
                aliases=names[1:],
            )

            return cls

        return decorator

    def get_class(self, name: str) -> type[ConstraintBase]:
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
    ) -> tuple[type[ConstraintBase], dict[str, Any]] | ConstraintBase | None:
        """Resolve a name, strict mapping, or class.

        Supported inputs:
            str:
                Registered name or alias.
            Mapping:
                {"type": registered_name, "params": {...}}
            class:
                Concrete subclass of this backend's base class.

        Return None, an existing instance, or a class with parameters.
        """
        if spec is None:
            return None

        if isinstance(spec, self._base_class):
            return spec
        
        if isinstance(spec, str):
            return self.get_class(spec), {}

        if isinstance(spec, Mapping):
            if not all(isinstance(key, str) for key in spec):
                raise ConstraintConfigurationError(
                    "Constraint specification keys must be strings."
                )

            unknown = set(spec) - _SPEC_KEYS
            if unknown:
                raise ConstraintConfigurationError(
                    "Unknown constraint specification keys: "
                    + ", ".join(sorted(unknown))
                )

            name = spec.get("type")
            if not isinstance(name, str) or not name.strip():
                raise ConstraintConfigurationError(
                    "constraint_spec.type must be a non-empty string."
                )

            raw_params = spec.get("params", {})
            if not isinstance(raw_params, Mapping):
                raise ConstraintConfigurationError(
                    "constraint_spec.params must be a mapping."
                )

            if not all(
                isinstance(key, str)
                for key in raw_params
            ):
                raise ConstraintConfigurationError(
                    "constraint_spec.params keys must be strings."
                )

            params = dict(raw_params)
            reserved = set(params) & _RESERVED_PARAMS

            if reserved:
                raise ConstraintConfigurationError(
                    "Constraint parameters reserved for prepared data: "
                    + ", ".join(sorted(reserved))
                )

            return self.get_class(name), params

        if isinstance(spec, type):
            return self._validate_class(spec), {}

        raise TypeError(
                "Constraint specification must be None, a registered name, "
                "a mapping with 'type' and optional 'params', "
                "a concrete constraint class, or an instance of "
                f"{self._base_class.__name__}."
                )