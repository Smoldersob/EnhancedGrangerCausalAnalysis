"""Utilities shared by common-order lag selectors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ....core.lag_config import LagSpec
from .incremental import BaseIncrementalLagSelector


def validate_order_selector_input(
    data: pd.DataFrame,
    outputs: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Reuse the existing selector input validation.

    This delegates validation only, not incremental search.
    """
    return BaseIncrementalLagSelector._validate_input(
        data,
        outputs,
    )


def common_order_lag_spec(
    predictors: tuple[str, ...],
    outputs: tuple[str, ...],
    order: int,
    *,
    use_lag_zero: bool,
) -> LagSpec:
    """Expand one selected order into explicit per-pair lag sets."""
    positive_lags = tuple(range(1, order + 1))

    return {
        output: {
            predictor: (
                (0,) + positive_lags
                if use_lag_zero and predictor != output
                else positive_lags
            )
            for predictor in predictors
        }
        for output in outputs
    }