from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ...core.exceptions import (
    StationarityNotFittedError,
    StationarityTestError,
)
from .base import BaseStationarityEnforcer, StationarityFitResult
from .tests import (
    apply_differencing,
    static_adfuller_order,
    static_kpss_order,
)
from .registry import register_stationarity_enforcer

class BaseDifferencingEnforcer(BaseStationarityEnforcer):
    """Shared workflow for test-guided differencing."""

    method: str

    def __init__(
        self,
        *,
        max_differencing_order: int = 5,
        alpha: float = 0.05,
        maxlag: int | None = None,
        exclude: Sequence[str] | None = None,
        dropna: bool = True,
    ) -> None:
        if (
            isinstance(max_differencing_order, bool)
            or not isinstance(max_differencing_order, Integral)
            or max_differencing_order < 0
        ):
            raise StationarityTestError(
                "max_differencing_order must be a non-negative integer."
            )

        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not np.isfinite(alpha)
            or not 0 < alpha < 1
        ):
            raise StationarityTestError(
                "alpha must be a finite number in (0, 1)."
            )

        if maxlag is not None and (
            isinstance(maxlag, bool)
            or not isinstance(maxlag, Integral)
            or maxlag < 0
        ):
            raise StationarityTestError(
                "maxlag must be a non-negative integer or None."
            )

        if not isinstance(dropna, bool):
            raise StationarityTestError("dropna must be a boolean.")

        if exclude is None:
            excluded = ()
        elif isinstance(exclude, (str, bytes)) or not isinstance(
            exclude, Sequence
        ):
            raise StationarityTestError(
                "exclude must be a sequence of column names or None."
            )
        else:
            excluded = tuple(exclude)

        if not all(isinstance(name, str) for name in excluded):
            raise StationarityTestError(
                "exclude must contain only strings."
            )

        if len(excluded) != len(set(excluded)):
            raise StationarityTestError(
                "exclude must not contain duplicate names."
            )

        self.max_differencing_order = int(max_differencing_order)
        self.alpha = float(alpha)
        self.maxlag = None if maxlag is None else int(maxlag)
        self.exclude = excluded
        self.dropna = dropna

        self.feature_names_in_: tuple[str, ...] = ()
        self.differencing_orders_: dict[str, int] = {}
        self.fit_result_: StationarityFitResult | None = None

    @staticmethod
    def _validate_data(
        data_list: Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        if isinstance(data_list, pd.DataFrame) or not isinstance(
            data_list, Sequence
        ):
            raise StationarityTestError(
                "Expected a non-empty sequence of DataFrames."
            )

        if len(data_list) == 0:
            raise StationarityTestError("The dataset sequence is empty.")

        result: list[pd.DataFrame] = []
        reference_columns: tuple[str, ...] | None = None

        for index, frame in enumerate(data_list):
            if not isinstance(frame, pd.DataFrame):
                raise StationarityTestError(
                    f"Dataset {index} is not a DataFrame."
                )

            if frame.empty:
                raise StationarityTestError(
                    f"Dataset {index} has no rows or columns."
                )

            if frame.columns.has_duplicates:
                raise StationarityTestError(
                    f"Dataset {index} contains duplicate column names."
                )

            columns = tuple(frame.columns)
            if not all(isinstance(name, str) for name in columns):
                raise StationarityTestError(
                    "Stationarity enforcers require string column names."
                )

            if reference_columns is None:
                reference_columns = columns
            elif columns != reference_columns:
                raise StationarityTestError(
                    "All datasets must have identical columns "
                    "in identical order."
                )

            try:
                numeric = frame.astype(float).copy()
            except (TypeError, ValueError) as exc:
                raise StationarityTestError(
                    f"Dataset {index} must contain numeric values."
                ) from exc

            if not np.isfinite(numeric.to_numpy()).all():
                raise StationarityTestError(
                    f"Dataset {index} contains missing or infinite values. "
                    "Impute missing values before enforcing stationarity."
                )

            result.append(numeric)

        return result

    @abstractmethod
    def _estimate_order(self, series: pd.Series) -> int:
        raise NotImplementedError

    def fit(
        self,
        data_list: Sequence[pd.DataFrame],
    ) -> BaseDifferencingEnforcer:
        self.feature_names_in_ = ()
        self.differencing_orders_ = {}
        self.fit_result_ = None

        frames = self._validate_data(data_list)
        columns = tuple(frames[0].columns)

        unknown = sorted(set(self.exclude) - set(columns))
        if unknown:
            raise StationarityTestError(
                f"Unknown excluded variables: {unknown}."
            )

        excluded = set(self.exclude)
        orders = {name: 0 for name in columns}

        for frame in frames:
            for name in columns:
                if name in excluded:
                    continue

                order = self._estimate_order(frame[name])

                if (
                    isinstance(order, bool)
                    or not isinstance(order, Integral)
                    or not 0 <= order <= self.max_differencing_order
                ):
                    raise StationarityTestError(
                        f"Invalid differencing order for {name!r}: {order!r}."
                    )

                orders[name] = max(orders[name], int(order))

        self.feature_names_in_ = columns
        self.differencing_orders_ = orders
        self.fit_result_ = StationarityFitResult(
            differencing_orders=orders.copy(),
            method=self.method,
            max_differencing_order=self.max_differencing_order,
            alpha=self.alpha,
        )
        return self

    def transform(
        self,
        data_list: Sequence[pd.DataFrame],
    ) -> list[pd.DataFrame]:
        if self.fit_result_ is None:
            raise StationarityNotFittedError(
                "Enforcer is not fitted. Call fit() first."
            )

        frames = self._validate_data(data_list)

        if tuple(frames[0].columns) != self.feature_names_in_:
            raise StationarityTestError(
                "Input columns do not match the fitted columns and order."
            )

        transformed: list[pd.DataFrame] = []

        for index, frame in enumerate(frames):
            output = frame.copy()

            for name, order in self.differencing_orders_.items():
                output[name] = apply_differencing(output[name], order)

            if self.dropna:
                output = output.dropna()

            if output.empty:
                raise StationarityTestError(
                    f"No observations remain in dataset {index} "
                    "after differencing."
                )

            transformed.append(output)

        return transformed

@register_stationarity_enforcer("adf")
class ADFStationarityEnforcer(BaseDifferencingEnforcer):
    """Select differencing orders using the existing ADF helper."""

    method = "adf"

    def _estimate_order(self, series: pd.Series) -> int:
        return static_adfuller_order(
            series=series,
            max_differencing_order=self.max_differencing_order,
            maxlag=self.maxlag,
            alpha=self.alpha,
        )

@register_stationarity_enforcer("kpss")
class KPSSStationarityEnforcer(BaseDifferencingEnforcer):
    """Select differencing orders using the existing KPSS helper."""

    method = "kpss"

    def _estimate_order(self, series: pd.Series) -> int:
        return static_kpss_order(
            series=series,
            max_differencing_order=self.max_differencing_order,
            maxlag=self.maxlag,
            alpha=self.alpha,
        )