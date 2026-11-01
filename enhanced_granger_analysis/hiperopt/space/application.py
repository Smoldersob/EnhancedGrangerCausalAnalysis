"""Apply candidate updates to declarative configuration containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import SearchCandidate
from .parser import SearchSpaceParser


def _copy_containers(value: Any) -> Any:
    """Copy declaration containers and arrays, preserving opaque objects."""
    if isinstance(value, Mapping):
        return {
            key: _copy_containers(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_copy_containers(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(
                "Object arrays are not supported in search declarations."
            )
        return value.copy()

    return value

def apply_candidate(
    base_configuration: Mapping[str, Any],
    candidate: SearchCandidate,
) -> dict[str, Any]:
    """Return configuration with validated candidate updates applied."""
    if not isinstance(base_configuration, Mapping):
        raise TypeError("Base configuration must be a mapping.")

    if not isinstance(candidate, SearchCandidate):
        raise TypeError("candidate must be a SearchCandidate.")

    declarations = {}

    for update in candidate.updates:
        path = update.path

        if (
            not isinstance(path, tuple)
            or len(path) < 2
            or any(
                not isinstance(part, str)
                or not part
                or part != part.strip()
                or "." in part
                for part in path
            )
        ):
            raise ValueError(f"Invalid candidate path: {path!r}.")

        name = ".".join(path)

        if name in declarations:
            raise ValueError(
                f"Repeated candidate path: {name!r}."
            )

        declarations[name] = {"choices": [None]}

    # Reuse path, reserved-field and prefix-conflict validation.
    SearchSpaceParser().parse(declarations)

    result = _copy_containers(base_configuration)

    for update in candidate.updates:
        current = result

        for position, part in enumerate(update.path[:-1]):
            prefix = ".".join(update.path[:position + 1])

            if part not in current:
                raise ValueError(
                    f"Missing intermediate configuration section: "
                    f"{prefix!r}."
                )

            child = current[part]

            if not isinstance(child, dict):
                raise TypeError(
                    f"Configuration section {prefix!r} must be "
                    "a mapping. Ready instances cannot be updated "
                    "through parameter paths."
                )

            current = child

        current[update.path[-1]] = _copy_containers(update.value)

    return result