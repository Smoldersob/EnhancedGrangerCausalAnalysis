from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ...core.exceptions import (
    StationarityNotFittedError,
    StationarityTestError,
)
from ...utilities.validation import validate_dataframe_list
from .tests import apply_differencing, static_adfuller_order, static_kpss_order


@dataclass
class StationarityFitResult:
    differencing_orders: Dict[str, int]
    test_name: str
    max_differencing_order: int
    alpha: float


class StationarityTransformer:
    """Fit and apply stationarity differencing on a list of datasets.

    The transformer searches the best differencing order in range
    ``[0, max_differencing_order]`` for each feature and then applies it
    consistently to all datasets.
    """

    def __init__(
        self,
        max_differencing_order: int = 5,
        test_name: str = "adf",
        alpha: float = 0.05,
        non_stationary: Optional[Sequence[str]] = None,
        dropna: bool = True,
    ) -> None:
        if max_differencing_order < 0:
            raise StationarityTestError("max_differencing_order must be >= 0")
        if test_name not in {"adf", "kpss"}:
            raise StationarityTestError("test_name must be one of: 'adf', 'kpss'")

        self.max_differencing_order = int(max_differencing_order)
        self.test_name = test_name
        self.alpha = float(alpha)
        self.non_stationary = set(non_stationary or [])
        self.dropna = dropna

        self.feature_names_in_: List[str] = []
        self.differencing_orders_: Dict[str, int] = {}
        self.fit_result_: Optional[StationarityFitResult] = None

    def _validate_data_list(self, data_list: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
        validated, _ = validate_dataframe_list(
            data_list,
            require_same_columns=True,
            require_same_shape=True,
            allow_superset_columns=True,
            copy=True,
        )
        return validated

    def _stationarity_order(self, series: pd.Series) -> int:
        if self.test_name == "kpss":
            return static_kpss_order(
                series=series,
                maxlag=self.max_differencing_order,
                alpha=self.alpha,
            )
        return static_adfuller_order(
            series=series,
            maxlag=self.max_differencing_order,
            alpha=self.alpha,
        )

    def fit_stationarity(self, data_list: Sequence[pd.DataFrame]) -> "StationarityTransformer":
        """Estimate per-feature differencing orders for provided datasets."""
        data_list_common_columns = self._validate_data_list(data_list)
        self.feature_names_in_ = list(data_list_common_columns[0].columns)

        fitted_orders = {name: 0 for name in self.feature_names_in_}
        for data in data_list_common_columns:
            for name in self.feature_names_in_:
                if name in self.non_stationary:
                    continue
                order = self._stationarity_order(data[name])
                fitted_orders[name] = max(fitted_orders[name], order)

        self.differencing_orders_ = fitted_orders
        self.fit_result_ = StationarityFitResult(
            differencing_orders=fitted_orders.copy(),
            test_name=self.test_name,
            max_differencing_order=self.max_differencing_order,
            alpha=self.alpha,
        )
        return self

    def transform(self, data_list: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
        """Apply fitted differencing orders to every dataset."""
        if not self.differencing_orders_:
            raise StationarityNotFittedError(
                "Transformer is not fitted. Call fit_stationarity first."
            )

        data_list_common_columns = self._validate_data_list(data_list)
        transformed_data = []

        for data in data_list_common_columns:
            data_out = data.copy()
            for name, order in self.differencing_orders_.items():
                data_out[name] = apply_differencing(data_out[name], order)
            if self.dropna:
                data_out = data_out.dropna()
            transformed_data.append(data_out)

        return transformed_data

    def fit_transform(self, data_list: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
        """Fit differencing orders and return transformed datasets."""
        self.fit_stationarity(data_list)
        return self.transform(data_list)

    def prepare_static(
        self,
        data_list: Sequence[pd.DataFrame],
        causes: Optional[Sequence[str]] = None,
        effects: Optional[Sequence[str]] = None,
    ):
        """Backward-compatible API used by legacy code paths."""
        transformed = self.fit_transform(data_list)

        causes = list(causes or self.feature_names_in_)
        effects = list(effects or self.feature_names_in_)
        columns_id = [self.feature_names_in_.index(cause) for cause in causes]
        nrows = len(effects)

        return nrows, columns_id, transformed


__all__ = [
    "StationarityFitResult",
    "StationarityTransformer",
]