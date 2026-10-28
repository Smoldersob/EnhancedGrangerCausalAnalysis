"""Historical VAR-based common-order heuristic."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from ....core.lag_config import LagSpec
from .base import BaseLagSelector
from .incremental import _integer
from .order_utils import (
    common_order_lag_spec,
    validate_order_selector_input,
)
from ..registry import register_lag_selector


@register_lag_selector(
    "naive_var",
    "var",
    "varlagselector",
)
class NaiveVARLagSelector(BaseLagSelector):
    """Historical VAR order selection with failed candidates skipped.

    For every candidate p, fit VAR(p) using statsmodels and obtain
    AIC, BIC, HQIC, and FPE.

    Choose each criterion's minimizing order. Return their maximum
    when maximum=True, otherwise their minimum.

    All supplied columns participate in the VAR system. outputs only
    restricts the returned lag specification.

    Lag 0 for cross-variable pairs, when requested, is added by policy
    after selection. It is not evaluated as a contemporaneous term.
    """

    def __init__(
        self,
        max_lag: int | None = None,
        *,
        center: bool = True,
        use_lag_zero: bool = False,
        min_lag: int = 1,
        maximum: bool = True,
    ) -> None:
        self.max_lag = (
            None
            if max_lag is None
            else _integer(
                max_lag,
                "max_lag",
                minimum=0,
            )
        )
        self.min_lag = _integer(
            min_lag,
            "min_lag",
            minimum=0,
        )

        if (
            self.max_lag is not None
            and self.min_lag > self.max_lag
        ):
            raise ValueError(
                "min_lag cannot exceed max_lag."
            )

        for name, value in (
            ("center", center),
            ("use_lag_zero", use_lag_zero),
            ("maximum", maximum),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool.")

        self.center = center
        self.use_lag_zero = use_lag_zero
        self.maximum = maximum

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outputs: Sequence[str] | None = None,
    ) -> LagSpec:
        names, targets, values = validate_order_selector_input(
            data,
            outputs,
        )

        for name in (
            "selected_order_",
            "best_orders_",
            "lag_spec_",
            "feature_names_in_",
            "output_names_",
        ):
            self.__dict__.pop(name, None)

        if len(names) < 2:
            raise ValueError(
                "NaiveVARLagSelector requires at least two variables."
            )

        max_lag = self.max_lag

        if max_lag is None:
            max_lag = 10
            warnings.warn(
                "max_lag not specified; defaulting to 10 "
                "for compatibility with the historical selector.",
                UserWarning,
                stacklevel=2,
            )

        if self.min_lag > max_lag:
            raise ValueError(
                "min_lag cannot exceed the effective max_lag."
            )

        if len(values) <= max_lag:
            raise ValueError(
                "The dataset must contain more rows than max_lag."
            )

        processed = values.copy()

        if self.center:
            processed -= processed.mean(
                axis=0,
                keepdims=True,
            )

        model = VAR(processed)

        rows = []
        failures = []

        for order in range(self.min_lag, max_lag + 1):
            try:
                with np.errstate(
                    over="raise",
                    invalid="raise",
                    divide="raise",
                ):
                    fitted = model.fit(order)

                    scores = {
                        "AIC": float(fitted.aic),
                        "BIC": float(fitted.bic),
                        "HQIC": float(fitted.hqic),
                        "FPE": float(fitted.fpe),
                    }

                if not all(
                    np.isfinite(value)
                    for value in scores.values()
                ):
                    raise FloatingPointError(
                        "At least one VAR criterion is non-finite."
                    )

            except (
                np.linalg.LinAlgError,
                ValueError,
                FloatingPointError,
                OverflowError,
                ZeroDivisionError,
            ) as exc:
                message = f"{type(exc).__name__}: {exc}"

                rows.append({
                    "order": order,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "HQIC": np.nan,
                    "FPE": np.nan,
                    "valid": False,
                    "error": message,
                })
                failures.append({
                    "order": order,
                    "reason": message,
                })

            else:
                rows.append({
                    "order": order,
                    **scores,
                    "valid": True,
                    "error": None,
                })

        self.order_scores_ = (
            pd.DataFrame(rows).set_index("order")
        )
        self.diagnostics_ = {
            "failed_orders": failures,
            "effective_max_lag": max_lag,
        }

        criteria = ["AIC", "BIC", "HQIC", "FPE"]

        valid_scores = self.order_scores_.loc[
            self.order_scores_["valid"],
            criteria,
        ]

        if valid_scores.empty:
            raise ValueError(
                "No VAR order produced finite values for all four "
                "criteria. Inspect order_scores_ and diagnostics_."
            )

        best_orders = {
            criterion: int(
                valid_scores[criterion].idxmin()
            )
            for criterion in criteria
        }

        selected_order = (
            max(best_orders.values())
            if self.maximum
            else min(best_orders.values())
        )

        result = common_order_lag_spec(
            names,
            targets,
            selected_order,
            use_lag_zero=self.use_lag_zero,
        )

        self.selected_order_ = selected_order
        self.best_orders_ = best_orders
        self.feature_names_in_ = names
        self.output_names_ = targets
        self.lag_spec_ = result

        return result


# Python import compatibility.
VARLagSelector = NaiveVARLagSelector