from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import isabstract
from typing import Any, Generic, TypeVar, cast

from ..models.protocol import GrangerModelProtocol
from ..registry import Registry


ModelT = TypeVar("ModelT", bound=GrangerModelProtocol)
ModelClass = TypeVar("ModelClass", bound=type)


_REQUIRED_METHODS = (
    "rebuild",
    "fit",
    "predict",
    "get_weights",
    "set_weights",
    "omit_variables",
    "reset_components",
)

_SPEC_KEYS = frozenset({"type", "params"})


class ModelClassRegistry(Generic[ModelT]):
    """Model class registry and specification resolver.

    Does not construct models and does not require a common base class.
    """

    def __init__(
        self,
        *,
        label: str,
    ) -> None:
        self._registry: Registry[type[ModelT]] = Registry(
            label=label,
        )

    def _validate_class(
        self,
        cls: Any,
    ) -> type[ModelT]:
        if not isinstance(cls, type):
            raise TypeError(
                "Only model classes may be registered."
            )

        if isabstract(cls) or getattr(cls, "_is_protocol", False):
            raise TypeError(
                "An abstract class or protocol cannot be selected as a model."
            )

        missing = [
            name
            for name in _REQUIRED_METHODS
            if not callable(getattr(cls, name, None))
        ]

        if missing:
            raise TypeError(
                f"Model class {cls.__name__!r} lacks required methods: "
                + ", ".join(missing)
            )

        return cast(type[ModelT], cls)

    def _validate_instance(
        self,
        instance: Any,
    ) -> ModelT:
        missing = [
            name
            for name in _REQUIRED_METHODS
            if not callable(getattr(instance, name, None))
        ]

        if missing:
            raise TypeError(
                "Model instance lacks required methods: "
                + ", ".join(missing)
            )

        backend = getattr(instance, "backend", None)
        if not isinstance(backend, str) or not backend.strip():
            raise TypeError(
                "Model instance must expose a non-empty backend identifier."
            )

        if not isinstance(getattr(instance, "is_fitted", None), bool):
            raise TypeError(
                "Model instance must expose a boolean is_fitted property."
            )

        return cast(ModelT, instance)

    def register(
        self,
        *names: str,
    ) -> Callable[[ModelClass], ModelClass]:
        """Register a class; the first name is canonical."""
        if not names:
            raise ValueError(
                "Model registration requires at least one name."
            )

        def decorator(
            cls: ModelClass,
        ) -> ModelClass:
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
    ) -> type[ModelT]:
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

    def validate_instance(self, instance: Any) -> ModelT:
        """Validate an existing model without resolving a specification.

        Checks required members, not complete signature/type compatibility.
        """
        if isinstance(instance, type):
            raise TypeError(
                "Expected a model instance, not a model class."
            )

        return self._validate_instance(instance)


    def resolve_spec(
        self,
        spec: Any,
    ) -> tuple[type[ModelT] | ModelT | None, dict[str, Any]]:
        """Resolve a specification without constructing a model.

        Always returns (target, params).

        target:
            None, a concrete model class or a validated model instance.

        params:
            Constructor parameters for a class.
            Empty for None and existing instances.
        """
        if spec is None:
            return None, {}

        if isinstance(spec, str):
            return self.get_class(spec), {}

        if isinstance(spec, type):
            return self._validate_class(spec), {}

        if isinstance(spec, Mapping):
            if any(not isinstance(key, str) for key in spec):
                raise TypeError(
                    "Model specification keys must be strings."
                )

            unknown = set(spec) - _SPEC_KEYS
            if unknown:
                raise ValueError(
                    "Unknown model specification keys: "
                    + ", ".join(sorted(unknown))
                )

            name = spec.get("type")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "model_spec.type must be a non-empty string."
                )

            params = spec.get("params", {})
            if not isinstance(params, Mapping):
                raise TypeError(
                    "model_spec.params must be a mapping."
                )

            if any(not isinstance(key, str) for key in params):
                raise TypeError(
                    "Model parameter names must be strings."
                )

            return self.get_class(name), dict(params)

        return self.validate_instance(spec), {}