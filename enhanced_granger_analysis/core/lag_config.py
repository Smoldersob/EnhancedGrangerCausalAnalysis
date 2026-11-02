"""Shared lag configuration and preparation-result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd


LagSpec: TypeAlias = dict[str, dict[str, tuple[int, ...]]]

LagFeature: TypeAlias = tuple[str, int]

Aggregation: TypeAlias = Literal[
    "union",
    "intersection",
    "frequency",
]


@dataclass(frozen=True)
class LagEngineConfig:
    """Configuration for final lag preparation.

    max_lag defines the fixed baseline when no selector is supplied.
    Automatic search settings belong to the selector.
    """

    max_lag: int = 12
    use_lag_zero: bool = False

    aggregation: Aggregation = "union"
    selection_frequency: float = 0.5

    predictor_overrides: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict
    )

    pair_overrides: Mapping[
        str,
        Mapping[str, tuple[int, ...]],
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class LagSegmentInfo:
    """Source correspondence for one independently prepared dataset.

    prepared_row_slice refers to concatenated rows before any subsequent
    pipeline sampling.
    """

    dataset_index: int
    original_row_count: int
    dropped_leading_rows: int
    prepared_row_slice: slice
    retained_index: pd.Index


@dataclass(frozen=True)
class LagFittedConfig:
    """Fitted lag layout and selection provenance, without prepared data.

    Returned by :meth:`LagEngine.fit` after selection, aggregation,
    overrides, and layout construction. The same layout is later used by
    :meth:`LagEngine.transform` to build lagged matrices from any frames
    that share the fitted schema.
    """

    output_names: tuple[str, ...]
    active_predictors: tuple[str, ...]
    removed_predictors: tuple[str, ...]

    feature_predictors: tuple[str, ...]
    feature_lags: np.ndarray
    block_boundaries: np.ndarray
    mask: np.ndarray

    final_lags: LagSpec
    aggregated_lags: LagSpec
    selection_results: tuple[LagSpec, ...]


@dataclass(frozen=True)
class LaggedData:
    """Lagged matrices built from one fitted layout.

    ``segments`` records the source correspondence of each independently
    prepared dataset, including the retained source index labels of the
    prepared rows.
    """

    X: np.ndarray
    y: np.ndarray
    segments: tuple[LagSegmentInfo, ...]


@dataclass(frozen=True)
class LagPreparationResult:
    """Prepared data, final feature metadata, and selection provenance.

    Compatibility container composed of :class:`LagFittedConfig` and
    :class:`LaggedData`; prefer the fit/transform pair directly.
    """

    X: np.ndarray
    y: np.ndarray

    output_names: tuple[str, ...]
    active_predictors: tuple[str, ...]
    removed_predictors: tuple[str, ...]

    feature_predictors: tuple[str, ...]
    feature_lags: np.ndarray
    block_boundaries: np.ndarray
    mask: np.ndarray

    final_lags: LagSpec
    aggregated_lags: LagSpec
    selection_results: tuple[LagSpec, ...]

    segments: tuple[LagSegmentInfo, ...]