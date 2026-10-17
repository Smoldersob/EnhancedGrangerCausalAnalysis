"""Named component registries independent of component construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar


T = TypeVar("T")


class RegistryError(ValueError):
    """Base error raised by component registries."""


class InvalidRegistryNameError(RegistryError):
    """A component name is invalid."""


class DuplicateRegistrationError(RegistryError):
    """A name or alias is already registered or repeated."""


class UnknownComponentError(RegistryError):
    """A requested component is not registered."""


def normalize_registry_name(name: str) -> str:
    """Validate and normalize a component name.

    Names are case-insensitive and ignore surrounding whitespace.
    Internal spaces, underscores and hyphens remain distinct.
    """
    if not isinstance(name, str):
        raise InvalidRegistryNameError(
            f"Component name must be a string, got {type(name).__name__}."
        )

    normalized = name.strip().lower()
    if not normalized:
        raise InvalidRegistryNameError("Component name must not be empty.")

    return normalized


class Registry(Generic[T]):
    """Map canonical names and aliases to components.

    The registry does not construct, clone or validate components.
    Registration is intended for application setup, not concurrent mutation.
    """

    def __init__(self, label: str = "component") -> None:
        self.label = label
        self._values: dict[str, T] = {}
        self._canonical_by_name: dict[str, str] = {}

    def register(
        self,
        name: str,
        value: T,
        *,
        aliases: Iterable[str] = (),
    ) -> T:
        """Register a component and return the unchanged value.

        All names are checked before the registry is modified.
        """
        if isinstance(aliases, (str, bytes)):
            raise InvalidRegistryNameError(
                "aliases must be an iterable of names, not a single string."
            )

        names = tuple(
            normalize_registry_name(item)
            for item in (name, *aliases)
        )

        if len(names) != len(set(names)):
            raise DuplicateRegistrationError(
                f"Repeated {self.label} names after normalization: {names!r}."
            )

        conflicts = sorted(
            item for item in names if item in self._canonical_by_name
        )
        if conflicts:
            raise DuplicateRegistrationError(
                f"Already registered {self.label} names: "
                f"{', '.join(conflicts)}."
            )

        canonical = names[0]
        self._values[canonical] = value
        self._canonical_by_name.update(
            (item, canonical) for item in names
        )
        return value

    def decorator(self, *names: str) -> Callable[[T], T]:
        """Return a decorator; the first name is canonical."""
        if not names:
            raise InvalidRegistryNameError(
                "Registration requires at least one name."
            )

        def decorate(value: T) -> T:
            return self.register(
                names[0],
                value,
                aliases=names[1:],
            )

        return decorate

    def canonical_name(self, name: str) -> str:
        """Return the canonical name for a registered name or alias."""
        normalized = normalize_registry_name(name)

        try:
            return self._canonical_by_name[normalized]
        except KeyError as exc:
            available = ", ".join(
                self.available(include_aliases=True)
            ) or "(none)"

            raise UnknownComponentError(
                f"Unknown {self.label} {name!r}. "
                f"Available names: {available}."
            ) from exc

    def resolve(self, name: str) -> T:
        """Return a registered value without invoking it."""
        return self._values[self.canonical_name(name)]

    def available(
        self,
        *,
        include_aliases: bool = False,
    ) -> tuple[str, ...]:
        """Return sorted canonical names, optionally including aliases."""
        source = (
            self._canonical_by_name
            if include_aliases
            else self._values
        )
        return tuple(sorted(source))

    def aliases_for(self, name: str) -> tuple[str, ...]:
        """Return aliases of a component, excluding its canonical name."""
        canonical = self.canonical_name(name)
        return tuple(sorted(
            alias
            for alias, target in self._canonical_by_name.items()
            if target == canonical and alias != canonical
        ))

    def copy(self) -> Registry[T]:
        """Copy registry mappings without copying registered components."""
        result = Registry[T](label=self.label)
        result._values = self._values.copy()
        result._canonical_by_name = self._canonical_by_name.copy()
        return result

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False

        try:
            normalized = normalize_registry_name(name)
        except InvalidRegistryNameError:
            return False

        return normalized in self._canonical_by_name

    def __len__(self) -> int:
        """Return the number of canonical components, excluding aliases."""
        return len(self._values)