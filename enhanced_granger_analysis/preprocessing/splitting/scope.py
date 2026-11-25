"""Immutable source-row boundaries authorized to influence fitting."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...core.exceptions import DataValidationError
from ...preprocessing.lag.artifacts import LaggedData


@dataclass(frozen=True)
class FitSegment:
    dataset_index: int
    row_start: int
    row_stop: int
    labels: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.dataset_index, bool)
            or isinstance(self.row_start, bool)
            or isinstance(self.row_stop, bool)
            or self.dataset_index < 0
            or self.row_start < 0
            or self.row_stop <= self.row_start
        ):
            raise ValueError("FitSegment requires non-negative, non-empty positional bounds.")
        object.__setattr__(self, "labels", tuple(self.labels))


@dataclass(frozen=True)
class FitScope:
    """Selected contiguous source ranges; ``whole_sample`` is explicit."""

    segments: tuple[FitSegment, ...]
    whole_sample: bool = False

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("FitScope must contain at least one source segment.")
        if any(not isinstance(segment, FitSegment) for segment in segments):
            raise TypeError("FitScope.segments must contain FitSegment values.")
        object.__setattr__(self, "segments", segments)

    @classmethod
    def all_data(cls, frames: Sequence[pd.DataFrame]) -> "FitScope":
        segments = tuple(
            FitSegment(index, 0, len(frame), tuple(frame.index))
            for index, frame in enumerate(frames)
            if len(frame)
        )
        return cls(segments=segments, whole_sample=True)

    def select_frames(
        self,
        frames: Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        selected: list[pd.DataFrame] = []
        for segment in self.segments:
            if segment.dataset_index >= len(frames):
                raise DataValidationError(
                    "FitScope refers to a dataset outside the current input."
                )
            frame = frames[segment.dataset_index]
            if segment.labels:
                selected.append(frame.loc[frame.index.isin(segment.labels)].copy())
            else:
                if segment.row_stop > len(frame):
                    raise DataValidationError(
                        "FitScope refers to rows outside the current input."
                    )
                selected.append(frame.iloc[segment.row_start:segment.row_stop].copy())
        if not selected or any(frame.empty for frame in selected):
            raise DataValidationError("FitScope contains an empty training segment.")
        return selected

    def bind(self, frames: Sequence[pd.DataFrame]) -> "FitScope":
        """Resolve positional ranges to stable source labels before transforms."""
        bound: list[FitSegment] = []
        for segment in self.segments:
            if segment.dataset_index >= len(frames):
                raise DataValidationError(
                    "FitScope refers to a dataset outside the current input."
                )
            frame = frames[segment.dataset_index]
            if segment.row_stop > len(frame):
                raise DataValidationError(
                    "FitScope refers to rows outside the current input."
                )
            if not self.whole_sample and not frame.index.is_unique:
                raise DataValidationError(
                    "A restricted FitScope requires unique source index labels "
                    "to preserve row identity across preprocessing transforms."
                )
            labels = tuple(frame.index[segment.row_start:segment.row_stop])
            if segment.labels and segment.labels != labels:
                raise DataValidationError(
                    "FitScope labels do not match the bound source row range."
                )
            bound.append(replace(segment, labels=labels))
        return FitScope(tuple(bound), whole_sample=self.whole_sample)

    def prepared_row_indices(
        self,
        lagged: LaggedData,
        source_frames: Sequence[pd.DataFrame],
        *,
        history: int,
    ) -> np.ndarray:
        """Return rows whose targets and lag history are inside this scope."""
        if self.whole_sample:
            return np.arange(lagged.X.shape[0], dtype=np.intp)
        selected: list[np.ndarray] = []
        for result_segment in lagged.segments:
            dataset_index = result_segment.dataset_index
            if dataset_index >= len(source_frames):
                raise DataValidationError("Lagged data refers to an unknown source dataset.")
            source_index = source_frames[dataset_index].index
            positions = source_index.get_indexer(result_segment.retained_index)
            if np.any(positions < 0):
                raise DataValidationError(
                    "Prepared rows cannot be mapped to the source FitScope."
                )
            local = np.zeros(positions.size, dtype=bool)
            for allowed in self.segments:
                if allowed.dataset_index != dataset_index:
                    continue
                local |= (
                    (positions >= allowed.row_start + history)
                    & (positions < allowed.row_stop)
                )
            if local.any():
                start = result_segment.prepared_row_slice.start
                selected.append(start + np.flatnonzero(local))
        if not selected:
            raise DataValidationError(
                "FitScope has no prepared rows with sufficient in-scope history."
            )
        return np.concatenate(selected).astype(np.intp, copy=False)


__all__ = ["FitScope", "FitSegment"]
