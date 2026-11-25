from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .feature_layout import validate_names

RemovalReason = Literal[
    "constant", "duplicate", "no_selected_lags", "structural_exclusion", "algebraic_redundancy",
]


@dataclass(frozen=True)
class RemovedVariable:
    """A signal removed from named analytical roles, not necessarily all roles."""

    name: str
    reason: RemovalReason
    removed_roles: tuple[Literal["effect", "cause", "tested_cause"], ...]
    retained_variable: Optional[str] = None

    def __post_init__(self) -> None:
        validate_names((self.name,), "Removed variable")
        roles = validate_names(self.removed_roles, "Removed roles")
        if not roles or not set(roles) <= {"effect", "cause", "tested_cause"}:
            raise ValueError("removed_roles must contain known analytical roles.")
        if self.reason not in {
            "constant", "duplicate", "no_selected_lags", "structural_exclusion", "algebraic_redundancy",
        }:
            raise ValueError(f"Unknown removal reason: {self.reason!r}.")
        if self.retained_variable is not None:
            validate_names((self.name, self.retained_variable), "Removed variable and representative")
            if self.reason != "duplicate":
                raise ValueError("Only duplicate removal can name a retained equivalent variable.")
        object.__setattr__(self, "removed_roles", roles)


@dataclass(frozen=True)
class LagSegmentInfo:
    """Source correspondence in pre-sampling coordinates for one fragment.

    ``dropped_leading_rows`` counts lag-history trimming within that fragment,
    not rows removed in preceding transformations.
    """

    dataset_index: int
    original_row_count: int
    dropped_leading_rows: int
    prepared_row_slice: slice
    retained_index: pd.Index

    def __post_init__(self) -> None:
        for name in ("dataset_index", "original_row_count", "dropped_leading_rows"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        block = self.prepared_row_slice
        if (
            not isinstance(block, slice)
            or (block.step is not None and (
                isinstance(block.step, (bool, np.bool_))
                or not isinstance(block.step, (int, np.integer))
                or block.step != 1
            ))
            or any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                for v in (block.start, block.stop)
            )
            or not 0 <= block.start <= block.stop
        ):
            raise ValueError("prepared_row_slice must be a bounded non-negative unit slice.")
        if not isinstance(self.retained_index, pd.Index):
            raise TypeError("retained_index must be a pandas Index.")
        if len(self.retained_index) != block.stop - block.start:
            raise ValueError("retained_index must contain one label per segment row.")
        if self.dropped_leading_rows + len(self.retained_index) > self.original_row_count:
            raise ValueError("Segment correspondence exceeds original_row_count.")
        object.__setattr__(self, "prepared_row_slice", slice(int(block.start), int(block.stop)))
        object.__setattr__(self, "retained_index", self.retained_index.copy(deep=True))
