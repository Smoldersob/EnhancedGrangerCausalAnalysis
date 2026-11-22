"""Data passed to model weight initializers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


WeightArray = NDArray[Any]


@dataclass(frozen=True)
class InitializationContext:
    X: WeightArray
    y: WeightArray
    weights: tuple[WeightArray, ...]
    mask: WeightArray | None = None