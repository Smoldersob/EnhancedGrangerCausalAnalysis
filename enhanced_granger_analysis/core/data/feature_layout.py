from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FeatureLayout:
    """Shared description of prepared feature blocks and their lags."""

    causes: tuple[str, ...]
    cause_slices: Mapping[str, slice]
    feature_lags: tuple[int, ...]

    def __post_init__(self) -> None:
        causes = tuple(self.causes)
        if not causes or len(causes) != len(set(causes)):
            raise ValueError("causes must contain unique variable names.")

        if not isinstance(self.cause_slices, Mapping):
            raise TypeError("cause_slices must be a mapping.")
        if set(self.cause_slices) != set(causes):
            raise ValueError("cause_slices keys must match causes.")

        lags = tuple(self.feature_lags)
        if any(
            isinstance(lag, (bool, np.bool_))
            or not isinstance(lag, (int, np.integer))
            or lag < 0
            for lag in lags
        ):
            raise ValueError("feature_lags must contain non-negative integers.")

        coverage = np.zeros(len(lags), dtype=np.int64)
        slices: dict[str, slice] = {}
        for cause in causes:
            block = self.cause_slices[cause]
            if not isinstance(block, slice) or block.step not in (None, 1):
                raise ValueError("Cause slices must be unit-step slices.")
            if (
                not isinstance(block.start, (int, np.integer))
                or isinstance(block.start, (bool, np.bool_))
                or not isinstance(block.stop, (int, np.integer))
                or isinstance(block.stop, (bool, np.bool_))
            ):
                raise ValueError("Cause slices must have explicit integer bounds.")
            start, stop = int(block.start), int(block.stop)
            if not 0 <= start < stop <= len(lags):
                raise ValueError("Invalid cause slice.")
            coverage[start:stop] += 1
            slices[cause] = slice(start, stop)

        if not np.all(coverage == 1):
            raise ValueError("Cause slices must cover every feature exactly once.")

        object.__setattr__(self, "causes", causes)
        object.__setattr__(self, "feature_lags", tuple(int(lag) for lag in lags))
        object.__setattr__(self, "cause_slices", MappingProxyType(slices))
