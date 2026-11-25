from dataclasses import dataclass

import numpy as np

from ...core.data.provenance import LagSegmentInfo
from ...core.lags.specification import LagSpec
from .space import LagSelectionSpace


@dataclass(frozen=True)
class LagFittedConfig:
    """Fitted lag layout and selection provenance."""

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
    selection_space: LagSelectionSpace | None = None
    overridden_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LaggedData:
    """Lagged matrices and source correspondence."""

    X: np.ndarray
    y: np.ndarray
    segments: tuple[LagSegmentInfo, ...]


@dataclass(frozen=True)
class LagPreparationResult:
    """Compatibility container for the combined lag prepare() operation."""

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
    selection_space: LagSelectionSpace | None = None
    overridden_pairs: tuple[tuple[str, str], ...] = ()
