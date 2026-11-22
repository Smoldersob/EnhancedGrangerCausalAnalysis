from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True, eq=False)
class RegularizationLayout:
    """Canonical coefficient layout for regularization."""

    cause_slices: Mapping[str, slice]
    output_names: tuple[str, ...]
    feature_lags: np.ndarray
    base_mask: np.ndarray

    def __post_init__(self) -> None:
        output_names = tuple(self.output_names)
        if not output_names or any(
            not isinstance(name, str) or not name for name in output_names
        ):
            raise ValueError("Output names must be non-empty strings.")
        if len(set(output_names)) != len(output_names):
            raise ValueError("Output names must be unique.")

        lags = np.asarray(self.feature_lags)
        if lags.ndim != 1 or lags.size == 0:
            raise ValueError("feature_lags must be a non-empty 1D array.")
        if lags.dtype.kind not in "iu":
            raise TypeError("feature_lags must contain integers.")
        if np.any(lags < 0):
            raise ValueError("feature_lags must be non-negative.")
        lags = np.array(lags, dtype=np.int64, copy=True)

        if not isinstance(self.cause_slices, Mapping):
            raise TypeError("cause_slices must be a mapping.")
        cause_slices: dict[str, slice] = {}
        coverage = np.zeros(len(lags), dtype=np.int64)
        for name, block in self.cause_slices.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Cause names must be non-empty strings.")
            if not isinstance(block, slice) or block.step not in (None, 1):
                raise ValueError("Cause slices must have unit step.")
            start, stop = block.start, block.stop
            if (
                not isinstance(start, (int, np.integer))
                or isinstance(start, (bool, np.bool_))
                or not isinstance(stop, (int, np.integer))
                or isinstance(stop, (bool, np.bool_))
            ):
                raise TypeError("Cause slices need explicit integer bounds.")
            start, stop = int(start), int(stop)
            if not 0 <= start < stop <= len(lags):
                raise ValueError(f"Invalid feature range for cause {name!r}.")
            coverage[start:stop] += 1
            cause_slices[name] = slice(start, stop)
        if np.any(coverage != 1):
            raise ValueError("cause_slices must cover every feature exactly once.")

        mask = np.asarray(self.base_mask)
        expected = (len(output_names), len(lags))
        if mask.shape != expected:
            raise ValueError(f"base_mask must have shape {expected}.")
        if mask.dtype.kind not in "biuf":
            raise TypeError("base_mask must be boolean or numeric.")
        if not np.all(np.isfinite(mask)):
            raise ValueError("base_mask must contain finite values.")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("base_mask must contain only 0 and 1.")
        mask = np.array(mask, dtype=bool, copy=True)
        lags.setflags(write=False)
        mask.setflags(write=False)

        object.__setattr__(self, "cause_slices", MappingProxyType(cause_slices))
        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "feature_lags", lags)
        object.__setattr__(self, "base_mask", mask)

    @property
    def n_features(self) -> int:
        return len(self.feature_lags)

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return self.n_outputs, self.n_features
