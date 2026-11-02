from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .regularizer_spec import RegularizerSpec


def parse_regularizer_spec(
    spec: Any,
) -> RegularizerSpec | None:
    """
    Accepted forms:

        None
        "l1"
        L1Regularizer
        {"type": "l1", "params": {"l1": 0.01}}
        {"type": CustomRegularizer, "params": {...}}
        RegularizerSpec(...)
    """
    if spec is None:
        return None

    if isinstance(spec, RegularizerSpec):
        return spec

    if isinstance(spec, (str, type)):
        return RegularizerSpec(selector=spec)

    if not isinstance(spec, Mapping):
        raise TypeError(
            "regularizer_spec must be None, a name, a class, "
            "a RegularizerSpec, or a mapping with 'type' and 'params'."
        )

    unknown_keys = set(spec) - {"type", "params"}

    if unknown_keys:
        raise ValueError(
            "Unknown regularizer specification keys: "
            f"{sorted(unknown_keys, key=repr)!r}."
        )

    if "type" not in spec:
        raise ValueError(
            "Regularizer specification requires the 'type' field."
        )

    return RegularizerSpec(
        selector=spec["type"],
        params=spec.get("params", {}),
    )