"""Resolution of declarative lag-engine configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
from typing import Any

from ...core.lag_config import LagEngineConfig


def _resolve_explicit_lags(
    value: Any,
    *,
    context: str,
) -> tuple[int, ...]:
    """Convert a serialized explicit lag sequence to a tuple.

    Value validation is performed by LagEngine.
    No range expansion is performed here.
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{context} must be a list or tuple of explicit lag values."
        )

    return tuple(value)


def _resolve_predictor_overrides(
    value: Any,
    *,
    context: str,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")

    result: dict[str, tuple[int, ...]] = {}

    for predictor, lags in value.items():
        if not isinstance(predictor, str):
            raise TypeError(
                f"{context} keys must be predictor names."
            )

        result[predictor] = _resolve_explicit_lags(
            lags,
            context=f"{context}[{predictor!r}]",
        )

    return result


def resolve_lag_engine_config(
    spec: LagEngineConfig | Mapping[str, Any] | None = None,
) -> LagEngineConfig:
    """Resolve a configuration without constructing a LagEngine."""
    if spec is None:
        return LagEngineConfig()

    field_names = {
        item.name
        for item in fields(LagEngineConfig)
        if item.init
    }

    if isinstance(spec, LagEngineConfig):
        values = {
            name: deepcopy(getattr(spec, name))
            for name in field_names
        }
    elif isinstance(spec, Mapping):
        values = deepcopy(dict(spec))

        unknown = set(values).difference(field_names)
        if unknown:
            raise ValueError(
                "Unknown lag-engine configuration fields: "
                f"{sorted(unknown, key=repr)!r}. "
                "Use predictor_overrides and pair_overrides "
                "with explicit lag sets."
            )
    else:
        raise TypeError(
            "lag_config must be None, a LagEngineConfig, "
            "or a mapping."
        )

    if "predictor_overrides" in values:
        values["predictor_overrides"] = (
            _resolve_predictor_overrides(
                values["predictor_overrides"],
                context="predictor_overrides",
            )
        )

    if "pair_overrides" in values:
        raw_pairs = values["pair_overrides"]

        if not isinstance(raw_pairs, Mapping):
            raise TypeError(
                "pair_overrides must be a mapping indexed by output."
            )

        pair_overrides: dict[
            str,
            dict[str, tuple[int, ...]],
        ] = {}

        for output, predictor_lags in raw_pairs.items():
            if not isinstance(output, str):
                raise TypeError(
                    "pair_overrides keys must be output names."
                )

            pair_overrides[output] = (
                _resolve_predictor_overrides(
                    predictor_lags,
                    context=f"pair_overrides[{output!r}]",
                )
            )

        values["pair_overrides"] = pair_overrides

    return LagEngineConfig(**values)