"""Assignments of prepared rows; no splitter is needed to read them."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

from ..exceptions import DataValidationError

SplitStage = Literal["source", "prepared"]
TRAINING_PART = "training"


def _validate_row_count(n_rows: int) -> None:
    if (
        isinstance(n_rows, (bool, np.bool_))
        or not isinstance(n_rows, (int, np.integer))
        or n_rows < 0
    ):
        raise DataValidationError("n_rows must be a non-negative integer.")


@dataclass(frozen=True, eq=False)
class DataSplit:
    """Assign final prepared rows to named parts; unassigned rows are excluded."""

    stage: SplitStage
    splitter_name: str | None
    parts: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        parts = {}
        for name, indices in self.parts.items():
            if not isinstance(name, str) or not name.strip():
                raise DataValidationError("Split part names must be non-empty strings.")
            array = np.array(indices, copy=True)
            if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
                raise DataValidationError(f"Split part {name!r} must use a 1D integer array.")
            parts[name] = array
        for name in ("validation", "test"):
            parts.setdefault(name, np.empty(0, dtype=np.intp))
        for array in parts.values():
            array.setflags(write=False)
        object.__setattr__(self, "parts", MappingProxyType(parts))

    @classmethod
    def all_training(cls, n_rows: int) -> "DataSplit":
        _validate_row_count(n_rows)
        return cls("prepared", None, {TRAINING_PART: np.arange(n_rows, dtype=np.intp)})

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(self.parts)

    def indices_for(self, part: str, *, require_nonempty: bool = False) -> np.ndarray:
        if part not in self.parts:
            raise DataValidationError(
                f"Unknown split part {part!r}. Available parts: {self.part_names}."
            )
        indices = self.parts[part]
        if require_nonempty and indices.size == 0:
            raise DataValidationError(f"Split part {part!r} is empty.")
        return indices

    @property
    def training_indices(self) -> np.ndarray | None:
        return self.parts.get(TRAINING_PART)

    def validate(self, n_rows: int) -> "DataSplit":
        _validate_row_count(n_rows)
        if self.stage not in {"source", "prepared"}:
            raise DataValidationError(f"Unknown split stage: {self.stage!r}.")
        if TRAINING_PART not in self.parts:
            raise DataValidationError("A data split must define the 'training' part.")
        collected = []
        for name, indices in self.parts.items():
            if np.any(indices < 0) or np.any(indices >= n_rows):
                raise DataValidationError(f"Split part {name!r} contains out-of-range indices.")
            if np.any(indices[1:] <= indices[:-1]):
                raise DataValidationError(f"Split part {name!r} must be strictly increasing.")
            collected.append(indices.astype(np.uint64, copy=False))
        combined = np.concatenate(collected)
        if np.unique(combined).size != combined.size:
            raise DataValidationError("Split parts must not share prepared rows.")
        return self

    def to_dict(self, *, n_rows: int) -> dict:
        self.validate(n_rows)
        return {
            "stage": self.stage,
            "splitter": self.splitter_name,
            "index_space": "prepared_rows",
            "n_samples": int(n_rows),
            "parts": {name: indices.tolist() for name, indices in self.parts.items()},
        }
