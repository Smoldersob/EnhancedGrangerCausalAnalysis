from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ...data.feature_layout import FeatureLayout, binary_mask, validate_names


@dataclass(frozen=True, eq=False)
class RegularizationLayout:
    """Canonical coefficient layout for regularization."""

    cause_slices: Mapping[str, slice]
    output_names: tuple[str, ...]
    feature_lags: np.ndarray
    base_mask: np.ndarray
    components: tuple[str | None, ...] = ()
    feature_layout: FeatureLayout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        output_names = validate_names(self.output_names, "Output names")
        if not output_names:
            raise ValueError("At least one output is required.")
        layout = FeatureLayout.from_slices(self.cause_slices, self.feature_lags, self.components)
        lags = np.array(layout.feature_lags, dtype=np.int64)
        mask = binary_mask(self.base_mask, (len(output_names), len(lags)))
        lags.setflags(write=False)

        object.__setattr__(self, "cause_slices", layout.cause_slices)
        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "feature_lags", lags)
        object.__setattr__(self, "base_mask", mask)
        object.__setattr__(self, "components", layout.components)
        object.__setattr__(self, "feature_layout", layout)

    @property
    def n_features(self) -> int:
        return len(self.feature_lags)

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return self.n_outputs, self.n_features
