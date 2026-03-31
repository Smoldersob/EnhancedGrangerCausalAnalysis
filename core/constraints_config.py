from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np
from numpy.typing import NDArray

# User-facing relation key: (output_variable_name, input_variable_name)
RelationKey = Tuple[str, str]

# User-facing relation value:
# - 0 / False / None / 'zero' -> force zero relation
# - positive float/int -> enforce minimal abs-sum for relation
# - {'min_abs_sum': x} or {'force_abs_sum': x} -> same as above
RelationValue = float | int | bool | str | Mapping[str, float] | None

# Full relation mapping provided by users.
RelationMap = Mapping[RelationKey, RelationValue]


@dataclass(frozen=True)
class MinAbsSumRule:
	"""A processed rule for enforcing minimal absolute weight sum."""

	output_index: int
	feature_indices: Tuple[int, ...]
	min_abs_sum: float


@dataclass(frozen=True)
class ProcessedConstraintSpec:
	"""Tensor-agnostic, validated constraint specification."""

	mask: NDArray[np.float64]
	rules: Tuple[MinAbsSumRule, ...]
