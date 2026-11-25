from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def validate_names(values, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{label} must be a sequence of names, not a string.")
    names = tuple(values)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must contain unique names.")
    return names


def binary_mask(value, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"Mask must have shape {shape}.")
    if mask.dtype.kind not in "biuf":
        raise TypeError("Mask must be boolean or numeric.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must contain only 0 and 1.")
    result = np.array(mask, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FeatureIdentity:
    """Identify a prepared column by source, lag and optional representation component."""

    source: str
    lag: int
    component: str | None = None

    def __post_init__(self) -> None:
        validate_names((self.source,), "Feature source")
        if isinstance(self.lag, (bool, np.bool_)) or not isinstance(
            self.lag, (int, np.integer)
        ) or self.lag < 0:
            raise ValueError("Feature lag must be a non-negative integer.")
        if self.component is not None:
            validate_names((self.component,), "Feature component")
        object.__setattr__(self, "lag", int(self.lag))


@dataclass(frozen=True)
class FeatureLayout:
    """Shared description of prepared feature blocks and their lags."""

    causes: tuple[str, ...]
    cause_slices: Mapping[str, slice]
    feature_lags: tuple[int, ...]
    components: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        causes = validate_names(self.causes, "causes")
        if not causes:
            raise ValueError("At least one cause is required.")

        if not isinstance(self.cause_slices, Mapping):
            raise TypeError("cause_slices must be a mapping.")
        if set(self.cause_slices) != set(causes):
            raise ValueError("cause_slices keys must match causes.")

        lags = tuple(self.feature_lags)
        if isinstance(self.components, str):
            raise ValueError("components must be a sequence, not a string.")
        components = tuple(self.components) or (None,) * len(lags)
        if len(components) != len(lags):
            raise ValueError("components must contain one entry per feature.")

        start_expected = 0
        slices: dict[str, slice] = {}
        features = []
        for cause in causes:
            block = _normalize_slice(self.cause_slices[cause])
            if block.start != start_expected or block.stop > len(lags):
                raise ValueError("Cause slices must cover every feature in causes order.")
            start_expected = block.stop
            slices[cause] = block
            features.extend(
                FeatureIdentity(cause, lags[index], components[index])
                for index in range(block.start, block.stop)
            )

        if start_expected != len(lags):
            raise ValueError("Cause slices must cover every feature in causes order.")
        if len(set(features)) != len(features):
            raise ValueError("Feature identities must be unique.")

        object.__setattr__(self, "causes", causes)
        object.__setattr__(self, "feature_lags", tuple(feature.lag for feature in features))
        object.__setattr__(self, "cause_slices", MappingProxyType(slices))
        object.__setattr__(self, "components", components)

    @classmethod
    def from_slices(cls, cause_slices, feature_lags, components=()) -> "FeatureLayout":
        """Recover physical block order from a mapping without ordered semantics."""
        if not isinstance(cause_slices, Mapping):
            raise TypeError("cause_slices must be a mapping.")
        slices = {name: _normalize_slice(block) for name, block in cause_slices.items()}
        causes = tuple(sorted(slices, key=lambda name: slices[name].start))
        return cls(causes, slices, feature_lags, components)

    @property
    def features(self) -> tuple[FeatureIdentity, ...]:
        return tuple(
            FeatureIdentity(source, self.feature_lags[i], self.components[i])
            for source, block in self.cause_slices.items()
            for i in range(block.start, block.stop)
        )


def _normalize_slice(block) -> slice:
    if not isinstance(block, slice):
        raise TypeError("Each cause slice must be a slice.")
    if block.step is not None and (
        isinstance(block.step, (bool, np.bool_))
        or not isinstance(block.step, (int, np.integer))
        or block.step != 1
    ):
        raise ValueError("Cause slices must have unit step.")
    if any(
        isinstance(bound, (bool, np.bool_)) or not isinstance(bound, (int, np.integer))
        for bound in (block.start, block.stop)
    ):
        raise TypeError("Cause slices must have explicit integer bounds.")
    if not 0 <= block.start < block.stop:
        raise ValueError("Invalid cause slice.")
    return slice(int(block.start), int(block.stop))
