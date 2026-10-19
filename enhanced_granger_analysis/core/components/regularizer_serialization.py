from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .regularizer_parsing import parse_regularizer_spec
from .regularizer_preparation import prepare_regularization_layout
from .regularizer_spec import (
    PreparedRegularizerData,
    RegularizerSpec,
)


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist())

    if isinstance(value, np.generic):
        return _to_json_value(value.item())

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Non-finite numbers cannot be serialized."
            )
        return value

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(
                "Serialized mapping keys must be strings."
            )

        return {
            key: _to_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]

    raise TypeError(
        f"Unsupported serialized value type: {type(value).__name__}."
    )


def serialize_regularizer_spec(
    spec: RegularizerSpec | None,
    *,
    registered_name: str | None = None,
) -> dict[str, Any] | None:
    if spec is None:
        return None

    if not isinstance(spec, RegularizerSpec):
        raise TypeError("spec must be a RegularizerSpec or None.")

    if registered_name is not None:
        name = registered_name
    elif isinstance(spec.selector, str):
        name = spec.selector
    else:
        raise ValueError(
            "Serializing a class-based specification requires "
            "its registered_name."
        )

    if not isinstance(name, str) or not name.strip():
        raise ValueError("registered_name must be a non-empty string.")

    return {
        "type": name.strip(),
        "params": _to_json_value(spec.params),
    }


def deserialize_regularizer_spec(
    payload: Mapping[str, Any] | None,
) -> RegularizerSpec | None:
    if payload is None:
        return None

    if not isinstance(payload, Mapping):
        raise TypeError("Serialized specification must be a mapping.")

    if not isinstance(payload.get("type"), str):
        raise TypeError(
            "Serialized regularizer type must be a registry name."
        )

    return parse_regularizer_spec(_to_json_value(payload))


def serialize_regularizer_data(
    data: PreparedRegularizerData,
) -> dict[str, Any]:
    if not isinstance(data, PreparedRegularizerData):
        raise TypeError("data must be PreparedRegularizerData.")

    layout = data.layout

    return {
        "schema_version": 1,
        "layout": {
            "cause_slices": {
                name: [block.start, block.stop]
                for name, block in layout.cause_slices.items()
            },
            "output_names": list(layout.output_names),
            "feature_lags": layout.feature_lags.tolist(),
            "base_mask": layout.base_mask.tolist(),
        },
        "params": _to_json_value(data.params),
    }


def deserialize_regularizer_data(
    payload: Mapping[str, Any],
) -> PreparedRegularizerData:
    if not isinstance(payload, Mapping):
        raise TypeError("Serialized data must be a mapping.")

    expected_keys = {"schema_version", "layout", "params"}

    if set(payload) != expected_keys:
        raise ValueError(
            f"Serialized data must contain exactly {expected_keys!r}."
        )

    version = payload["schema_version"]

    if type(version) is not int or version != 1:
        raise ValueError("Unsupported regularizer data schema version.")

    raw_layout = payload["layout"]

    if not isinstance(raw_layout, Mapping):
        raise TypeError("Serialized layout must be a mapping.")

    expected_layout_keys = {
        "cause_slices",
        "output_names",
        "feature_lags",
        "base_mask",
    }

    if set(raw_layout) != expected_layout_keys:
        raise ValueError(
            "Serialized layout contains missing or unexpected fields."
        )

    raw_slices = raw_layout["cause_slices"]

    if not isinstance(raw_slices, Mapping):
        raise TypeError("Serialized cause_slices must be a mapping.")

    cause_slices = {}

    for name, bounds in raw_slices.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"Invalid serialized slice for cause {name!r}."
            )

        cause_slices[name] = slice(bounds[0], bounds[1])

    layout = prepare_regularization_layout(
        cause_slices=cause_slices,
        output_names=raw_layout["output_names"],
        feature_lags=raw_layout["feature_lags"],
        base_mask=raw_layout["base_mask"],
    )

    return PreparedRegularizerData(
        layout=layout,
        params=_to_json_value(payload["params"]),
    )