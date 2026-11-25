from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..data.feature_layout import FeatureLayout, validate_names


@dataclass(frozen=True, eq=False)
class ModelInputSpec:
    """Structural description of prepared model inputs and outputs."""

    n_features: int
    n_outputs: int
    causes: tuple[str, ...]
    effects: tuple[str, ...]
    cause_slices: Mapping[str, slice]
    feature_lags: tuple[int, ...]
    components: tuple[str | None, ...] = ()
    feature_layout: FeatureLayout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("n_features", "n_outputs"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))

        effects = validate_names(self.effects, "effects")
        if len(effects) != self.n_outputs:
            raise ValueError("The number of effects must match n_outputs.")
        layout = FeatureLayout(self.causes, self.cause_slices, self.feature_lags, self.components)
        if len(layout.feature_lags) != self.n_features:
            raise ValueError("feature_lags must contain one lag per input feature.")
        for name in ("causes", "cause_slices", "feature_lags", "components"):
            object.__setattr__(self, name, getattr(layout, name))
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "feature_layout", layout)
