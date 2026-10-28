"""Abstract interface for lag-selection algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from ....core.lag_config import LagSpec


class BaseLagSelector(ABC):
    """Define the common contract for lag-selection algorithms.

    A selector processes one independent time-series dataset and returns
    explicit lag sets for each requested output/predictor pair.

    Concrete implementations define their own constructor parameters,
    selection procedure, scoring rules, and optional diagnostics.

    Selectors do not aggregate results across datasets, apply user lag
    overrides, construct final feature matrices, or build coefficient masks.
    """

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        *,
        outputs: Sequence[str] | None = None,
    ) -> LagSpec:
        """Select lags for one independent time-series dataset.

        Parameters
        ----------
        data
            Time-series observations ordered chronologically.

            Rows represent observations and columns represent variables.
            Column names must be unique strings.

            All supplied columns are candidate predictors. An output's
            own column is also available as an autoregressive predictor.

            Concrete selectors may use the supplied variables jointly
            during selection, even when only some outputs are requested.

        outputs
            Names of variables for which lag specifications are requested.
            If None, return specifications for all columns in data.

            Names must be unique, present in data, and provided as a
            sequence rather than a single string. The sequence must not
            be empty.

        Returns
        -------
        LagSpec
            Dictionary indexed first by output name and then by predictor
            name. Each value is a tuple of actual selected lag values.

            The returned specification must satisfy these rules:

            - Include exactly the requested outputs.
            - Include every data column as a predictor for each output.
            - Preserve requested output order and data-column order.
            - Use sorted tuples of unique non-negative integer lags.
            - Use an empty tuple for an excluded predictor.
            - Interpret lag d as the predictor value at time t - d.
            - Never include lag 0 for an output's own predictor.

            Tuples contain explicit lag values, not range endpoints.
            For example, (1, 3) means lags 1 and 3, not lags 1 through 3.

        Notes
        -----
        Implementations must not modify the supplied DataFrame.

        Each call must select lags using only the supplied dataset;
        selections must not accumulate across previous calls.

        Implementations may retain diagnostics from the latest successful
        call, but downstream preparation must depend on the returned
        specification rather than implementation-specific attributes.

        A selector may reject unsupported data or infeasible selection
        settings with a descriptive exception. It must not silently
        substitute a different selection criterion.
        """
        raise NotImplementedError