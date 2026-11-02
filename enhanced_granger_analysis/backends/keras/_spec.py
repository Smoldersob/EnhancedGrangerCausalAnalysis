from collections.abc import Mapping
from typing import Any


def extract_typed_spec(
    spec: Any,
    *,
    context: str,
) -> tuple[str, dict[str, Any]]:
    if isinstance(spec, str):
        name = spec.strip()
        if not name:
            raise ValueError(f"{context}: component name must not be empty.")
        return name, {}

    if not isinstance(spec, Mapping):
        raise TypeError(
            f"{context}: expected a name or a mapping."
        )

    name = spec.get("type", spec.get("name", spec.get("kind")))
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{context}: a non-empty 'type' field is required."
        )

    raw_params = spec.get("params", {})
    if raw_params is None:
        raw_params = {}

    if not isinstance(raw_params, Mapping):
        raise TypeError(f"{context}: 'params' must be a mapping.")

    params = dict(raw_params)

    # Zachowanie dotychczasowego formatu z parametrami na górnym poziomie.
    for key, value in spec.items():
        if key not in {"type", "name", "kind", "params"}:
            params[key] = value

    if any(not isinstance(key, str) for key in params):
        raise TypeError(f"{context}: parameter names must be strings.")

    return name.strip(), params