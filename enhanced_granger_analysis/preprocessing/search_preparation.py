"""Resolution helpers for the independent model-search preparation context."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
from typing import Any

from .pipeline import DataPreparationConfig


def resolve_search_preparation(
    analysis_preparation: DataPreparationConfig,
    search_overrides: Mapping[str, Any] | None,
) -> DataPreparationConfig:
    """Return a new preparation declaration for HPO without fitting it."""
    if search_overrides is None:
        return replace(analysis_preparation)
    if not isinstance(search_overrides, Mapping):
        raise TypeError("search.preparation must be a mapping or None.")

    unknown = set(search_overrides) - {"splitter", "random_state"}
    if unknown:
        raise ValueError(
            "Unsupported search preparation overrides: "
            f"{sorted(unknown, key=repr)!r}."
        )

    values: dict[str, Any] = {}
    for name in ("splitter", "random_state"):
        if name in search_overrides:
            values[name] = search_overrides[name]
    return replace(analysis_preparation, **values)


__all__ = ["resolve_search_preparation"]
