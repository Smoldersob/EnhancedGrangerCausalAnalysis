"""Contracts for splitting prepared data into named parts.

Splitters come in two stages:

- ``source``: configured before preprocessing. The splitter chooses the
  source frames used to fit preprocessing decisions (imputation,
  redundancy, stationarity, lag selection) and later assigns prepared
  rows to parts through recorded row correspondence.
- ``prepared``: configured after lag preparation and before scaling.
  Rows are complete lagged samples; the splitter assigns them directly.

Both stages produce the same final artifact: a :class:`DataSplit`
mapping part names to positional indices into the concatenated prepared
matrices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    ClassVar, 
    Literal, 
    Mapping, 
    Sequence, 
)
from types import MappingProxyType

import numpy as np
import pandas as pd

from ...core.exceptions import DataValidationError
from ...core.lag_config import LaggedData

SplitStage = Literal["source", "prepared"]

#: Conventional part name used when fitting downstream components.
TRAINING_PART = "training"


@dataclass(frozen=True)
class DataSplit:
    """Assign prepared rows to named parts.

    Empty validation and test parts are allowed.
    Unassigned rows remain excluded from all parts.
    """

    stage: SplitStage
    splitter_name: str | None
    parts: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        """Copy indices and add missing evaluation parts."""
        parts = {}

        for name, indices in self.parts.items():
            if not isinstance(name, str) or not name.strip():
                raise DataValidationError(
                    "Split part names must be non-empty strings."
                )

            array = np.array(indices, copy=True)

            if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
                raise DataValidationError(
                    f"Split part {name!r} must use a 1D integer array."
                )

            array.setflags(write=False)
            parts[name] = array

        for name in ("validation", "test"):
            if name not in parts:
                array = np.empty(0, dtype=np.intp)
                array.setflags(write=False)
                parts[name] = array

        object.__setattr__(
            self,
            "parts",
            MappingProxyType(parts),
        )

    @classmethod
    def all_training(cls, n_rows: int) -> "DataSplit":
        """Assign every prepared row to training."""
        if (
            isinstance(n_rows, (bool, np.bool_))
            or not isinstance(n_rows, (int, np.integer))
            or n_rows < 0
        ):
            raise DataValidationError(
                "n_rows must be a non-negative integer."
            )

        return cls(
            stage="prepared",
            splitter_name=None,
            parts={
                TRAINING_PART: np.arange(n_rows, dtype=np.intp),
            },
        ).validate(n_rows)

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(self.parts)

    def indices_for(
        self,
        part: str,
        *,
        require_nonempty: bool = False,
    ) -> np.ndarray:
        """Return positional indices for one part."""
        if part not in self.parts:
            raise DataValidationError(
                f"Unknown split part {part!r}. "
                f"Available parts: {self.part_names}."
            )

        indices = self.parts[part]

        if require_nonempty and indices.size == 0:
            raise DataValidationError(
                f"Split part {part!r} is empty."
            )

        return indices

    @property
    def training_indices(self) -> np.ndarray | None:
        """Return training indices, if the part exists."""
        return self.parts.get(TRAINING_PART)

    def validate(self, n_rows: int) -> "DataSplit":
        """Validate bounds, ordering and disjointness."""
        if (
            isinstance(n_rows, (bool, np.bool_))
            or not isinstance(n_rows, (int, np.integer))
            or n_rows < 0
        ):
            raise DataValidationError(
                "n_rows must be a non-negative integer."
            )

        if self.stage not in {"source", "prepared"}:
            raise DataValidationError(
                f"Unknown split stage: {self.stage!r}."
            )

        if TRAINING_PART not in self.parts:
            raise DataValidationError(
                "A data split must define the 'training' part."
            )

        collected = []

        for name, array in self.parts.items():
            if np.any(array < 0) or np.any(array >= n_rows):
                raise DataValidationError(
                    f"Split part {name!r} contains out-of-range indices."
                )

            if np.any(array[1:] <= array[:-1]):
                raise DataValidationError(
                    f"Split part {name!r} must be strictly increasing."
                )

            collected.append(array)

        combined = np.concatenate(collected)

        if np.unique(combined).size != combined.size:
            raise DataValidationError(
                "Split parts must not share prepared rows."
            )

        return self

    def to_dict(self, *, n_rows: int) -> dict:
        """Return a serializable snapshot of prepared-row assignments."""
        self.validate(n_rows)

        return {
            "stage": self.stage,
            "splitter": self.splitter_name,
            "index_space": "prepared_rows",
            "n_samples": int(n_rows),
            "parts": {
                name: indices.tolist()
                for name, indices in self.parts.items()
            },
        }


@dataclass(frozen=True)
class SourceBlock:
    """One contiguous row block of one source dataset assigned to a part.

    ``row_start``/``row_stop`` are positional bounds within the dataset
    as it entered preprocessing (half-open interval).
    """

    dataset_index: int
    part: str
    row_start: int
    row_stop: int


class BaseDataSplitter(ABC):
    """Interface implemented by all registered data splitters."""

    stage: ClassVar[SplitStage]

    @abstractmethod
    def split(self, lagged: LaggedData) -> DataSplit:
        """Assign prepared rows to named parts."""
        raise NotImplementedError


class SourceDataSplitter(BaseDataSplitter):
    """Base class for splitters decided before preprocessing.

    Row correspondence between source rows and prepared rows is kept
    through dataset index labels, so the plan stays valid when
    transformations drop rows (for example ``dropna`` after
    differencing). Source datasets must therefore have unique index
    labels.
    """

    stage: ClassVar[SplitStage] = "source"

    def __init__(self) -> None:
        self._indexes: list[pd.Index] | None = None
        self._blocks_by_dataset: list[tuple[SourceBlock, ...]] | None = None

    def fit(self, datasets: Sequence[pd.DataFrame]) -> "SourceDataSplitter":
        """Record dataset geometry and build the block plan."""
        frames = self._validate_datasets(datasets)

        self._indexes = [frame.index for frame in frames]
        self._blocks_by_dataset = [
            tuple(blocks) for blocks in self._build_blocks(frames)
        ]

        self._validate_plan(frames)

        if not any(
            block.part == TRAINING_PART
            for blocks in self._blocks_by_dataset
            for block in blocks
        ):
            raise DataValidationError(
                "The data splitter defines no training rows; "
                "preprocessing decisions cannot be fitted."
            )

        return self

    @abstractmethod
    def _build_blocks(
        self,
        datasets: Sequence[pd.DataFrame],
    ) -> Sequence[Sequence[SourceBlock]]:
        """Return the ordered blocks of each dataset."""
        raise NotImplementedError

    def _part_names(self) -> tuple[str, ...]:
        assert self._blocks_by_dataset is not None
        names: list[str] = []
        for blocks in self._blocks_by_dataset:
            for block in blocks:
                if block.part not in names:
                    names.append(block.part)
        return tuple(names)

    def select_for_fitting(
        self,
        datasets: Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        """Return the training blocks used to fit preprocessing decisions.

        Each contiguous training block is returned as its own frame;
        non-contiguous blocks are never concatenated into a new series.
        """
        if self._indexes is None or self._blocks_by_dataset is None:
            raise RuntimeError(
                f"{type(self).__name__}.fit() must be called before "
                "select_for_fitting()."
            )

        if len(datasets) != len(self._indexes):
            raise DataValidationError(
                "The dataset sequence changed after the splitter was "
                "fitted; call fit() again on the current data."
            )

        training_frames: list[pd.DataFrame] = []

        for dataset_index, (frame, blocks) in enumerate(
            zip(datasets, self._blocks_by_dataset)
        ):
            if not isinstance(frame, pd.DataFrame):
                raise DataValidationError(
                    f"Dataset {dataset_index} is not a DataFrame."
                )

            for block in blocks:
                if block.part != TRAINING_PART:
                    continue

                labels = self._indexes[dataset_index][
                    block.row_start:block.row_stop
                ]

                training_frames.append(
                    frame.loc[frame.index.isin(labels)].copy()
                )

        if not training_frames:
            raise DataValidationError(
                "The active data splitter defines no training rows; "
                "preprocessing decisions cannot be fitted."
            )

        return training_frames

    def split(self, lagged: LaggedData) -> DataSplit:
        """Map prepared rows to parts through recorded row correspondence."""
        if self._indexes is None or self._blocks_by_dataset is None:
            raise RuntimeError(
                f"{type(self).__name__}.fit() must be called before "
                "split()."
            )

        part_names = self._part_names()
        collected: dict[str, list[np.ndarray]] = {
            part: [] for part in part_names
        }

        for segment in lagged.segments:
            dataset_index = segment.dataset_index

            if dataset_index >= len(self._blocks_by_dataset):
                raise DataValidationError(
                    "Lagged segments do not match the datasets used to "
                    "fit the splitter."
                )

            blocks = self._blocks_by_dataset[dataset_index]
            index = self._indexes[dataset_index]

            try:
                positions = index.get_indexer(segment.retained_index)
            except Exception as exc:
                raise DataValidationError(
                    "Prepared-row correspondence requires the source "
                    "index recorded by the splitter."
                ) from exc

            if np.any(positions < 0):
                raise DataValidationError(
                    "Prepared rows refer to source labels that the "
                    "splitter did not record; the data or index changed "
                    "after the splitter was fitted."
                )

            part_ids = self._part_of(dataset_index, positions)

            segment_start = segment.prepared_row_slice.start

            for block_index, block in enumerate(blocks):
                local_rows = np.flatnonzero(part_ids == block_index)

                if local_rows.size:
                    collected[block.part].append(
                        segment_start + local_rows
                    )

        parts: dict[str, np.ndarray] = {}

        for part in part_names:
            if collected[part]:
                parts[part] = np.sort(
                    np.concatenate(collected[part])
                ).astype(np.intp, copy=False)

        if not parts:
            raise DataValidationError(
                "The data splitter assigned no prepared rows to any part."
            )

        split = DataSplit(
            stage=self.stage,
            splitter_name=type(self).__name__,
            parts=parts,
        )

        return split.validate(lagged.X.shape[0])

    def _part_of(
        self,
        dataset_index: int,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Return the block index owning each source row position."""
        assert self._blocks_by_dataset is not None
        blocks = self._blocks_by_dataset[dataset_index]

        boundaries = np.asarray(
            [block.row_start for block in blocks] + [blocks[-1].row_stop],
            dtype=np.int64,
        )

        return np.searchsorted(boundaries, positions, side="right") - 1

    @staticmethod
    def _validate_datasets(
        datasets: Sequence[pd.DataFrame],
    ) -> tuple[pd.DataFrame, ...]:
        if isinstance(datasets, pd.DataFrame):
            raise DataValidationError(
                "Source splitters require a sequence of DataFrames, "
                "not one DataFrame."
            )

        frames = tuple(datasets)

        if not frames:
            raise DataValidationError(
                "Source splitters require at least one DataFrame."
            )

        for index, frame in enumerate(frames):
            if not isinstance(frame, pd.DataFrame):
                raise DataValidationError(
                    f"Dataset {index} is not a DataFrame."
                )

            if frame.empty:
                raise DataValidationError(f"Dataset {index} is empty.")

            if not frame.index.is_unique:
                raise DataValidationError(
                    f"Dataset {index} has duplicated index labels; "
                    "source splitters require unique labels so prepared "
                    "rows can be traced back to their source position. "
                    "Consider reset_index()."
                )

        return frames

    def _validate_plan(
        self,
        frames: Sequence[pd.DataFrame],
    ) -> None:
        assert self._blocks_by_dataset is not None

        for dataset_index, blocks in enumerate(self._blocks_by_dataset):
            if not blocks:
                raise DataValidationError(
                    f"Dataset {dataset_index} received no split blocks."
                )

            previous_stop = 0

            for block in blocks:
                if (
                    block.dataset_index != dataset_index
                    or block.row_start < previous_stop
                    or block.row_stop <= block.row_start
                    or block.row_stop > len(frames[dataset_index])
                ):
                    raise DataValidationError(
                        f"Invalid split blocks for dataset {dataset_index}."
                    )

                previous_stop = block.row_stop

            if previous_stop != len(frames[dataset_index]):
                raise DataValidationError(
                    f"Split blocks of dataset {dataset_index} do not "
                    "cover all rows. Unassigned trailing rows are not "
                    "supported by this splitter; parts must cover every "
                    "row."
                )


class PreparedDataSplitter(BaseDataSplitter):
    """Base class for splitters decided after lag preparation."""

    stage: ClassVar[SplitStage] = "prepared"

    def __init__(self) -> None:
        self._n_rows: int | None = None

    def fit(self, n_rows: int) -> "PreparedDataSplitter":
        """Record the prepared row count for geometry validation."""
        if not isinstance(n_rows, int) or n_rows <= 0:
            raise DataValidationError(
                "Prepared splitters require a positive row count."
            )

        self._n_rows = int(n_rows)
        return self

    def _ensure_fitted(self, n_rows: int) -> None:
        if self._n_rows is None:
            raise RuntimeError(
                f"{type(self).__name__}.fit() must be called before "
                "split()."
            )

        if self._n_rows != n_rows:
            raise DataValidationError(
                "The prepared row count changed after the splitter was "
                "fitted; call fit() again on the current data."
            )
