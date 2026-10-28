from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class StationarityFitResult:
    """Summary of fitted per-variable differencing orders."""

    differencing_orders: dict[str, int]
    method: str
    max_differencing_order: int
    alpha: float


class BaseStationarityEnforcer(ABC):
    """Contract for fitted stationarity preprocessing components."""

    @abstractmethod
    def fit(
        self,
        data_list: Sequence[pd.DataFrame],
    ) -> BaseStationarityEnforcer:
        """Estimate transformation parameters without modifying input data."""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        data_list: Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        """Apply fitted parameters without estimating them again."""
        raise NotImplementedError

    def fit_transform(
        self,
        data_list: Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        return self.fit(data_list).transform(data_list)