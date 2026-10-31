"""Built-in data splitters for both split stages.

Source-stage splitters (decided before preprocessing):

- ``dataset_level``: whole source datasets are assigned to parts.
- ``temporal_blocks``: every dataset is divided into contiguous
  chronological blocks, one per part, in the order the parts are given.

Prepared-stage splitters (decided after lag preparation, before
scaling). Both emit a warning: stationarity and lag selection were
fitted on the full sample, so validation under these splitters is not
fully independent of preprocessing:

- ``random_rows``: prepared rows are assigned to parts globally at
  random.
- ``stratified_random_rows``: the random assignment is performed within
  each source dataset, preserving part proportions per series.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ...core.exceptions import DataValidationError
from ...core.lag_config import LaggedData
from .base import (
    TRAINING_PART,
    DataSplit,
    PreparedDataSplitter,
    SourceBlock,
    SourceDataSplitter,
)
from .registry import register_data_splitter

_PREPARED_STAGE_WARNING = (
    "{name!r} splits completed lagged rows after stationarity and lag "
    "selection were fitted on the full sample. Validation under this "
    "split may be optimistic because preprocessing decisions already "
    "saw the validation rows (information leakage). Use a source-stage "
    "splitter for validation that is independent of preprocessing."
)


def _validate_parts(parts: Any) -> dict[str, Any]:
    if not isinstance(parts, Mapping) or not parts:
        raise DataValidationError(
            "Splitter parts must be a non-empty mapping of part name "
            "to fraction or count."
        )

    normalized: dict[str, Any] = {}

    for name, value in parts.items():
        if not isinstance(name, str) or not name.strip():
            raise DataValidationError(
                "Splitter part names must be non-empty strings."
            )

        normalized[name] = value

    return normalized


def _part_counts(
    n_rows: int,
    parts: Mapping[str, Any],
    *,
    full_coverage: bool,
) -> dict[str, int]:
    """Convert part fractions or counts into integer row counts.

    Fractional values are distributed with the largest-remainder rule.
    With ``full_coverage`` the counts must cover every row exactly;
    otherwise the counts only need to fit within the data.
    """
    values = list(parts.items())
    fractions = [
        value for _, value in values
        if isinstance(value, (float, np.floating))
    ]

    use_fractions = len(fractions) == len(values)

    if not use_fractions and not all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        for _, value in values
    ):
        raise DataValidationError(
            "Splitter part sizes must be either all fractions in (0, 1] "
            "or all integer counts, not mixed."
        )

    if use_fractions:
        total = float(np.sum([float(value) for _, value in values]))

        if full_coverage:
            if abs(total - 1.0) > 1e-9:
                raise DataValidationError(
                    "Splitter part fractions must sum to 1 when parts "
                    "must cover every row; "
                    f"got {total!r}."
                )
        else:
            if total > 1.0 + 1e-9 or total <= 0.0:
                raise DataValidationError(
                    "Splitter part fractions must sum to a value in "
                    f"(0, 1]; got {total!r}."
                )

        target = (
            n_rows
            if full_coverage
            else min(n_rows, int(round(total * n_rows)))
        )

        exact = np.asarray(
            [float(value) * n_rows for _, value in values]
        )
        counts = np.floor(exact).astype(np.int64)

        remainder = target - int(counts.sum())

        if remainder < 0 or remainder > len(values):
            raise DataValidationError(
                "Splitter part fractions do not yield a valid row "
                "assignment for this data size."
            )

        order = np.argsort(-(exact - counts), kind="stable")

        for position in range(remainder):
            counts[order[position]] += 1

        result = {
            name: int(count) for (name, _), count in zip(values, counts)
        }
    else:
        result = {
            name: int(value) for name, value in values
        }

        total = sum(result.values())

        if full_coverage:
            if total != n_rows:
                raise DataValidationError(
                    "Splitter part counts must sum to the number of "
                    f"rows ({n_rows}); got {total}."
                )
        else:
            if total > n_rows:
                raise DataValidationError(
                    "Splitter part counts exceed the number of rows "
                    f"({n_rows}); got {total}."
                )

    for name, count in result.items():
        if count <= 0:
            raise DataValidationError(
                f"Splitter part {name!r} receives no rows for this "
                "data size; adjust the part configuration."
            )

    return result


@register_data_splitter(
    "dataset_level",
    "by_dataset",
    "whole_datasets",
)
class DatasetLevelSplitter(SourceDataSplitter):
    """Assign whole source datasets to parts.

    Parts are supplied as a mapping of part name to either the number
    of datasets in that part (assigned in order) or an explicit
    sequence of dataset positions, for example::

        {"type": "dataset_level", "params": {"parts": {
            "training": 2,
            "validation": 1,
        }}}

        {"type": "dataset_level", "params": {"parts": {
            "training": [0, 2],
            "validation": [1],
        }}}

    Every dataset must be assigned to exactly one part, and at least
    one dataset must be training data.
    """

    def __init__(self, parts: Mapping[str, Any]) -> None:
        super().__init__()
        self.parts = _validate_parts(parts)

    def _build_blocks(
        self,
        datasets: Sequence[pd.DataFrame],
    ) -> list[list[SourceBlock]]:
        n_datasets = len(datasets)

        assignment: dict[int, str] = {}
        consumed: set[int] = set()

        for part, value in self.parts.items():
            if isinstance(value, (int, np.integer)) and not isinstance(
                value, bool
            ):
                pass
            elif isinstance(value, Sequence) and not isinstance(value, str):
                for dataset_index in value:
                    if not isinstance(dataset_index, (int, np.integer)):
                        raise DataValidationError(
                            f"Explicit dataset positions of part {part!r} "
                            "must be integers."
                        )
            else:
                raise DataValidationError(
                    f"Part {part!r} must map to a dataset count or an "
                    "explicit sequence of dataset positions."
                )

        for part, value in self.parts.items():
            if isinstance(value, (int, np.integer)) and not isinstance(
                value, bool
            ):
                available = [
                    index for index in range(n_datasets)
                    if index not in consumed
                ]

                if value < 1 or len(available) < value:
                    raise DataValidationError(
                        f"Part {part!r} requests {value} datasets, but "
                        f"only {len(available)} remain unassigned."
                    )

                for dataset_index in available[: int(value)]:
                    assignment[dataset_index] = part
                    consumed.add(dataset_index)
            else:
                for dataset_index in value:
                    dataset_index = int(dataset_index)

                    if not 0 <= dataset_index < n_datasets:
                        raise DataValidationError(
                            f"Part {part!r} references dataset position "
                            f"{dataset_index}, which does not exist."
                        )

                    if dataset_index in consumed:
                        raise DataValidationError(
                            f"Dataset position {dataset_index} is "
                            "assigned to more than one part."
                        )

                    assignment[dataset_index] = part
                    consumed.add(dataset_index)

        unassigned = [
            index for index in range(n_datasets)
            if index not in assignment
        ]

        if unassigned:
            raise DataValidationError(
                "Every dataset must be assigned to exactly one part; "
                f"unassigned dataset positions: {unassigned}."
            )

        return [
            [
                SourceBlock(
                    dataset_index=dataset_index,
                    part=assignment[dataset_index],
                    row_start=0,
                    row_stop=len(frame),
                )
            ]
            for dataset_index, frame in enumerate(datasets)
        ]


@register_data_splitter(
    "temporal_blocks",
    "source_temporal",
    "per_series_blocks",
)
class TemporalBlockSplitter(SourceDataSplitter):
    """Divide every dataset into contiguous chronological blocks.

    Parts are supplied as a mapping of part name to a row fraction or
    count. Blocks are laid out in the order the parts are given, so
    supply them chronologically, for example::

        {"type": "temporal_blocks", "params": {"parts": {
            "training": 0.8,
            "validation": 0.2,
        }}}

    With fractional sizes the parts must sum to 1; every row of every
    dataset is assigned. Training rows form a leading prefix of each
    dataset and are used to fit preprocessing decisions.
    """

    def __init__(self, parts: Mapping[str, Any]) -> None:
        super().__init__()
        self.parts = _validate_parts(parts)

    def _build_blocks(
        self,
        datasets: Sequence[pd.DataFrame],
    ) -> list[list[SourceBlock]]:
        blocks: list[list[SourceBlock]] = []

        for dataset_index, frame in enumerate(datasets):
            counts = _part_counts(
                len(frame),
                self.parts,
                full_coverage=True,
            )

            dataset_blocks: list[SourceBlock] = []
            position = 0

            for part, count in counts.items():
                dataset_blocks.append(
                    SourceBlock(
                        dataset_index=dataset_index,
                        part=part,
                        row_start=position,
                        row_stop=position + count,
                    )
                )
                position += count

            blocks.append(dataset_blocks)

        return blocks


@register_data_splitter("random_rows", "prepared_random")
class RandomRowSplitter(PreparedDataSplitter):
    """Globally random assignment of prepared rows to parts.

    Rows are complete lagged samples, so parts contain rows from any
    series. Stationarity and lag selection were fitted on the full
    sample; a warning is emitted because validation under this splitter
    is not fully independent of preprocessing.
    """

    def __init__(
        self,
        parts: Mapping[str, Any],
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.parts = _validate_parts(parts)
        self.random_state = random_state

        warnings.warn(
            _PREPARED_STAGE_WARNING.format(name="random_rows"),
            UserWarning,
            stacklevel=2,
        )

    def split(self, lagged: LaggedData) -> DataSplit:
        n_rows = lagged.X.shape[0]
        self._ensure_fitted(n_rows)

        counts = _part_counts(
            n_rows,
            self.parts,
            full_coverage=False,
        )

        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(n_rows)

        parts: dict[str, np.ndarray] = {}
        position = 0

        for part, count in counts.items():
            parts[part] = np.sort(
                order[position:position + count]
            ).astype(np.intp, copy=False)
            position += count

        return DataSplit(
            stage=self.stage,
            splitter_name="random_rows",
            parts=parts,
        ).validate(n_rows)


@register_data_splitter(
    "stratified_random_rows",
    "by_series_random",
    "per_dataset_random",
)
class StratifiedRandomRowSplitter(PreparedDataSplitter):
    """Random assignment of prepared rows within each source dataset.

    Part proportions are preserved within every series, then merged
    across series. Stationarity and lag selection were fitted on the
    full sample; a warning is emitted because validation under this
    splitter is not fully independent of preprocessing.
    """

    def __init__(
        self,
        parts: Mapping[str, Any],
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.parts = _validate_parts(parts)
        self.random_state = random_state

        if not all(
            isinstance(value, (float, np.floating))
            for value in self.parts.values()
        ):
            raise DataValidationError(
                "stratified_random_rows requires fractional part "
                "sizes; integer counts are ambiguous per series."
            )

        warnings.warn(
            _PREPARED_STAGE_WARNING.format(
                name="stratified_random_rows"
            ),
            UserWarning,
            stacklevel=2,
        )

    def split(self, lagged: LaggedData) -> DataSplit:
        n_rows = lagged.X.shape[0]
        self._ensure_fitted(n_rows)

        rng = np.random.default_rng(self.random_state)

        collected: dict[str, list[np.ndarray]] = {
            part: [] for part in self.parts
        }

        for segment in lagged.segments:
            segment_rows = np.arange(
                segment.prepared_row_slice.start,
                segment.prepared_row_slice.stop,
            )

            segment_n = segment_rows.size

            if segment_n < len(self.parts):
                continue

            counts = _part_counts(
                segment_n,
                self.parts,
                full_coverage=False,
            )

            order = rng.permutation(segment_rows)

            position = 0

            for part, count in counts.items():
                collected[part].append(
                    order[position:position + count]
                )
                position += count

        parts: dict[str, np.ndarray] = {}

        for part, row_groups in collected.items():
            if row_groups:
                parts[part] = np.sort(
                    np.concatenate(row_groups)
                ).astype(np.intp, copy=False)

        if not parts:
            raise DataValidationError(
                "The stratified splitter assigned no prepared rows; "
                "the series are too short for this part configuration."
            )

        if TRAINING_PART not in parts:
            raise DataValidationError(
                "The stratified splitter produced no training rows."
            )

        return DataSplit(
            stage=self.stage,
            splitter_name="stratified_random_rows",
            parts=parts,
        ).validate(n_rows)
