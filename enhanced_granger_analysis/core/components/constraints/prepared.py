from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...data.feature_layout import binary_mask
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
        expected = (len(output_names), self.n_features)
        try:
            mask = binary_mask(self.mask, expected)
        except (TypeError, ValueError) as exc:
            raise ConstraintConfigurationError(str(exc)) from exc
        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "rules", tuple(self.rules))
