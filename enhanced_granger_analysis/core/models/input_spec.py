from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, eq=False)
class ModelInputSpec:
    """Structural description of prepared model inputs and outputs.

    Contains no observations, fitted transformers or training state.

    cause_slices and feature_lags describe the columns of prepared X,
    not the original unlagged dataset.
    """

    n_features: int
    n_outputs: int

    causes: tuple[str, ...]
    effects: tuple[str, ...]

    cause_slices: Mapping[str, slice]
    feature_lags: tuple[int, ...]

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

        causes = tuple(self.causes)
        effects = tuple(self.effects)

        for name, values in (("causes", causes), ("effects", effects)):
            if any(
                not isinstance(value, str) or not value
                for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings.")

            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique names.")

        if not causes:
            raise ValueError("At least one cause is required.")

        if len(effects) != self.n_outputs:
            raise ValueError(
                "The number of effects must match n_outputs."
            )

        if not isinstance(self.cause_slices, Mapping):
            raise TypeError("cause_slices must be a mapping.")

        if set(self.cause_slices) != set(causes):
            raise ValueError(
                "cause_slices keys must match causes."
            )

        coverage = np.zeros(self.n_features, dtype=np.int64)
        blocks: dict[str, slice] = {}

        for cause in causes:
            block = self.cause_slices[cause]

            if not isinstance(block, slice):
                raise TypeError(
                    f"cause_slices[{cause!r}] must be a slice."
                )

            if block.step not in (None, 1):
                raise ValueError("Cause slices must have unit step.")

            for bound in (block.start, block.stop):
                if (
                    isinstance(bound, (bool, np.bool_))
                    or not isinstance(bound, (int, np.integer))
                ):
                    raise TypeError(
                        "Cause slices must have explicit integer bounds."
                    )

            start = int(block.start)
            stop = int(block.stop)

            if not 0 <= start < stop <= self.n_features:
                raise ValueError(
                    f"Invalid feature slice for cause {cause!r}."
                )

            blocks[cause] = slice(start, stop)
            coverage[start:stop] += 1

        if not np.all(coverage == 1):
            raise ValueError(
                "Cause slices must cover all features exactly once."
            )

        lags = tuple(self.feature_lags)

        if len(lags) != self.n_features:
            raise ValueError(
                "feature_lags must contain one lag per input feature."
            )

        if any(
            isinstance(lag, (bool, np.bool_))
            or not isinstance(lag, (int, np.integer))
            or lag < 0
            for lag in lags
        ):
            raise ValueError(
                "feature_lags must contain non-negative integers."
            )

        object.__setattr__(self, "causes", causes)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(
            self, "cause_slices", MappingProxyType(blocks)
        )
        object.__setattr__(
            self, "feature_lags", tuple(int(lag) for lag in lags)
        )