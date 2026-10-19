from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np


RegularizerSelector: TypeAlias = str | type


@dataclass(frozen=True, eq=False)
class RegularizerSpec:
    """Normalized selection of a regularizer and its parameters."""

    selector: RegularizerSelector
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selector = self.selector

        if isinstance(selector, str):
            selector = selector.strip()
            if not selector:
                raise ValueError("Regularizer name must not be empty.")
        elif not isinstance(selector, type):
            raise TypeError(
                "Regularizer selector must be a name or a class."
            )

        if not isinstance(self.params, Mapping):
            raise TypeError("Regularizer params must be a mapping.")

        if any(not isinstance(key, str) for key in self.params):
            raise TypeError("Regularizer parameter names must be strings.")

        object.__setattr__(self, "selector", selector)
        object.__setattr__(
            self,
            "params",
            MappingProxyType(deepcopy(dict(self.params))),
        )


@dataclass(frozen=True, eq=False)
class RegularizationLayout:
    """
    Metadata for a canonical coefficient matrix:

        shape == (n_outputs, n_features)

    feature_lags contains actual lags, not positions within cause blocks.
    """

    cause_slices: Mapping[str, slice]
    output_names: tuple[str, ...]
    feature_lags: np.ndarray
    base_mask: np.ndarray

    def __post_init__(self) -> None:
        output_names = tuple(self.output_names)

        if not output_names:
            raise ValueError("At least one output is required.")

        if any(
            not isinstance(name, str) or not name
            for name in output_names
        ):
            raise ValueError("Output names must be non-empty strings.")

        if len(set(output_names)) != len(output_names):
            raise ValueError("Output names must be unique.")

        lags = np.asarray(self.feature_lags)

        if lags.ndim != 1 or lags.size == 0:
            raise ValueError(
                "feature_lags must be a non-empty one-dimensional array."
            )

        if lags.dtype.kind not in "iu":
            raise TypeError("feature_lags must contain integers.")

        if np.any(lags < 0):
            raise ValueError("feature_lags must be non-negative.")

        if np.any(lags > np.iinfo(np.int64).max):
            raise ValueError("feature_lags exceed the supported range.")

        lags = np.array(lags, dtype=np.int64, copy=True)
        n_features = len(lags)

        if not isinstance(self.cause_slices, Mapping):
            raise TypeError("cause_slices must be a mapping.")

        cause_slices: dict[str, slice] = {}
        coverage = np.zeros(n_features, dtype=np.int64)

        for name, block in self.cause_slices.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Cause names must be non-empty strings.")

            if not isinstance(block, slice):
                raise TypeError(
                    f"cause_slices[{name!r}] must be a slice."
                )

            if block.step not in (None, 1):
                raise ValueError(
                    f"cause_slices[{name!r}] must have step 1."
                )

            start, stop = block.start, block.stop

            if (
                not isinstance(start, (int, np.integer))
                or isinstance(start, (bool, np.bool_))
                or not isinstance(stop, (int, np.integer))
                or isinstance(stop, (bool, np.bool_))
            ):
                raise TypeError(
                    f"cause_slices[{name!r}] needs explicit integer bounds."
                )

            start, stop = int(start), int(stop)

            if not 0 <= start < stop <= n_features:
                raise ValueError(
                    f"Invalid feature range for cause {name!r}: "
                    f"{start}:{stop}."
                )

            coverage[start:stop] += 1
            cause_slices[name] = slice(start, stop)

        if np.any(coverage != 1):
            raise ValueError(
                "cause_slices must cover every feature exactly once."
            )

        mask = np.asarray(self.base_mask)

        expected_shape = (len(output_names), n_features)

        if mask.shape != expected_shape:
            raise ValueError(
                f"base_mask must have shape {expected_shape}, "
                f"got {mask.shape}."
            )

        if mask.dtype.kind not in "biuf":
            raise TypeError("base_mask must be boolean or numeric.")

        if not np.all(np.isfinite(mask)):
            raise ValueError("base_mask must contain finite values.")

        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("base_mask must contain only 0 and 1.")

        mask = np.array(mask, dtype=bool, copy=True)

        lags.setflags(write=False)
        mask.setflags(write=False)

        object.__setattr__(
            self, "cause_slices", MappingProxyType(cause_slices)
        )
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

    def relation_indices(
        self,
        cause: str,
        effect: str,
        *,
        active_only: bool = True,
    ) -> tuple[int, np.ndarray]:
        """
        Return a canonical coefficient row and feature-column indices.

        These are coefficient indices, not observation indices.
        """
        if cause not in self.cause_slices:
            raise KeyError(f"Unknown cause: {cause!r}.")

        try:
            output_index = self.output_names.index(effect)
        except ValueError:
            raise KeyError(f"Unknown effect: {effect!r}.") from None

        block = self.cause_slices[cause]
        feature_indices = np.arange(
            block.start, block.stop, dtype=np.int64
        )

        if active_only:
            feature_indices = feature_indices[
                self.base_mask[output_index, feature_indices]
            ]

        return output_index, feature_indices


@dataclass(frozen=True, eq=False)
class PreparedRegularizerData:
    """The complete input passed to a regularizer constructor."""

    layout: RegularizationLayout
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, RegularizationLayout):
            raise TypeError(
                "layout must be a RegularizationLayout instance."
            )

        if not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping.")

        if any(not isinstance(key, str) for key in self.params):
            raise TypeError("Parameter names must be strings.")

        object.__setattr__(
            self,
            "params",
            MappingProxyType(deepcopy(dict(self.params))),
        )