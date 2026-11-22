from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...exceptions import ConstraintConfigurationError


@dataclass(frozen=True)
class PreparedMinAbsSumRule:
    output_index: int
    feature_indices: tuple[int, ...]
    min_abs_sum: float


@dataclass(frozen=True, eq=False)
class PreparedConstraintData:
    """Backend-independent prepared constraint data."""

    output_names: tuple[str, ...]
    n_features: int
    mask: NDArray[np.bool_]
    rules: tuple[PreparedMinAbsSumRule, ...]

    def __post_init__(self) -> None:
        output_names = tuple(self.output_names)
        mask = np.array(self.mask, dtype=bool, copy=True)
        expected = (len(output_names), self.n_features)
        if mask.shape != expected:
            raise ConstraintConfigurationError(
                f"mask shape {mask.shape} does not match {expected}"
            )
        mask.setflags(write=False)
        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "rules", tuple(self.rules))
