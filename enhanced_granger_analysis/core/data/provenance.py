from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class RemovedVariable:
    """A variable removed during redundant-column cleanup."""

    name: str
    reason: Literal["constant", "duplicate", "no_selected_lags"]
    removed_roles: tuple[Literal["effect", "cause", "tested_cause"], ...]
    retained_variable: Optional[str] = None


@dataclass(frozen=True)
class LagSegmentInfo:
    """Source correspondence for one lag-prepared dataset."""

    dataset_index: int
    original_row_count: int
    dropped_leading_rows: int
    prepared_row_slice: slice
    retained_index: pd.Index
