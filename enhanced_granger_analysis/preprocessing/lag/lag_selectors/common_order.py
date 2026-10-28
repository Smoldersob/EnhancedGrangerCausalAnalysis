"""Selection of one common lag order for all requested output equations."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ....core.lag_config import LagSpec
from .base import BaseLagSelector
from .incremental import (
    CandidateEvaluationError,
    _augment_intercept,
    _integer,
    _nonnegative_finite,
)
from .incremental import ICLagSelector, CVLagSelector
from .order_utils import (
    common_order_lag_spec,
    validate_order_selector_input,
)
from ..registry import register_lag_selector


def _validate_scoring_params(
    params: dict[str, Any],
    allowed: set[str],
) -> None:
    unknown = set(params).difference(allowed)

    if unknown:
        raise TypeError(
            "Unsupported scoring parameters: "
            f"{sorted(unknown)!r}."
        )


class BaseCommonOrderLagSelector(BaseLagSelector):
    """Evaluate every common order from 1 through max_lag.

    All candidates use the same response timestamps. Every supplied
    data column is a predictor; only requested outputs are scored.

    Cross-variable lag 0, when enabled, is included in every candidate.
    Own-variable lag 0 is never included.
    """

    def __init__(
        self,
        *,
        max_lag: int,
        use_lag_zero: bool,
        output_weights: Mapping[str, float] | None,
        scorer: ICLagSelector | CVLagSelector,
    ) -> None:
        self.max_lag = _integer(
            max_lag,
            "max_lag",
            minimum=1,
        )

        if not isinstance(use_lag_zero, bool):
            raise TypeError("use_lag_zero must be a bool.")

        if (
            output_weights is not None
            and not isinstance(output_weights, Mapping)
        ):
            raise TypeError(
                "output_weights must be a mapping or None."
            )

        self.use_lag_zero = use_lag_zero
        self.output_weights = (
            None
            if output_weights is None
            else dict(output_weights)
        )
        self._scorer = scorer

    @abstractmethod
    def _score_order(
        self,
        responses: np.ndarray,
        designs: tuple[np.ndarray, ...],
        weights: np.ndarray,
    ) -> float:
        raise NotImplementedError

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

        if len(values) <= self.max_lag:
            raise ValueError(
                "The dataset must contain more rows than max_lag."
            )

        for name in (
            "lag_spec_",
            "selected_order_",
            "feature_names_in_",
            "output_names_",
            "selection_start_",
            "n_selection_samples_",
            "output_weights_",
        ):
            self.__dict__.pop(name, None)

        weights = self._resolve_output_weights(targets)

        self._scorer.diagnostics_ = {
            "rank_deficient_fits": 0,
            "variance_floor_uses": 0,
            "failed_candidates": [],
        }
        self.diagnostics_ = {
            "failed_orders": [],
            "scoring": self._scorer.diagnostics_,
        }

        self.order_scores_ = pd.DataFrame(
            columns=["order", "score", "valid", "error"]
        )

        start = self.max_lag
        n_rows = len(values) - start

        name_to_index = {
            name: index
            for index, name in enumerate(names)
        }

        responses = values[
            start:,
            [name_to_index[name] for name in targets],
        ]

        first_lag = 0 if self.use_lag_zero else 1

        columns = {
            (name, lag): values[
                start - lag : len(values) - lag,
                index,
            ]
            for index, name in enumerate(names)
            for lag in range(first_lag, self.max_lag + 1)
        }

        rows: list[dict[str, Any]] = []
        best_order: int | None = None
        best_score: float | None = None

        for order in range(1, self.max_lag + 1):
            designs = tuple(
                self._build_design(
                    columns,
                    names,
                    target,
                    order,
                    n_rows,
                )
                for target in targets
            )

            try:
                with np.errstate(
                    over="raise",
                    invalid="raise",
                    divide="raise",
                ):
                    score = float(
                        self._score_order(
                            responses,
                            designs,
                            weights,
                        )
                    )

                if not np.isfinite(score):
                    raise CandidateEvaluationError(
                        "The aggregated score is non-finite."
                    )

            except (
                CandidateEvaluationError,
                np.linalg.LinAlgError,
                FloatingPointError,
                OverflowError,
            ) as exc:
                message = f"{type(exc).__name__}: {exc}"

                rows.append({
                    "order": order,
                    "score": np.nan,
                    "valid": False,
                    "error": message,
                })
                self.diagnostics_["failed_orders"].append({
                    "order": order,
                    "reason": message,
                })

            else:
                rows.append({
                    "order": order,
                    "score": score,
                    "valid": True,
                    "error": None,
                })

                # Ascending order + strict improvement preserves
                # the smaller order for numerically equivalent scores.
                if (
                    best_score is None
                    or self._scorer._strictly_better(
                        score,
                        best_score,
                    )
                ):
                    best_order = order
                    best_score = score

        self.order_scores_ = (
            pd.DataFrame(rows).set_index("order")
        )

        if best_order is None:
            raise ValueError(
                "No common order could be scored. "
                "Inspect order_scores_ and diagnostics_."
            )

        result = common_order_lag_spec(
            names,
            targets,
            best_order,
            use_lag_zero=self.use_lag_zero,
        )

        self.selected_order_ = best_order
        self.feature_names_in_ = names
        self.output_names_ = targets
        self.output_weights_ = dict(zip(targets, weights))
        self.selection_start_ = start
        self.n_selection_samples_ = n_rows
        self.lag_spec_ = result

        return result

    def _build_design(
        self,
        columns: dict[tuple[str, int], np.ndarray],
        predictors: tuple[str, ...],
        target: str,
        order: int,
        n_rows: int,
    ) -> np.ndarray:
        active_columns = []

        for predictor in predictors:
            first_lag = (
                0
                if self.use_lag_zero and predictor != target
                else 1
            )

            for lag in range(first_lag, order + 1):
                active_columns.append(
                    columns[(predictor, lag)]
                )

        return (
            np.column_stack(active_columns)
            if active_columns
            else np.empty((n_rows, 0), dtype=float)
        )

    def _resolve_output_weights(
        self,
        outputs: tuple[str, ...],
    ) -> np.ndarray:
        raw = (
            {}
            if self.output_weights is None
            else self.output_weights
        )

        unknown = set(raw).difference(outputs)
        if unknown:
            raise ValueError(
                "Weights reference unknown outputs: "
                f"{sorted(unknown, key=repr)!r}."
            )

        weights = np.asarray(
            [
                _nonnegative_finite(
                    raw.get(output, 1.0),
                    f"output_weights[{output!r}]",
                )
                for output in outputs
            ],
            dtype=float,
        )

        maximum = float(weights.max())
        if maximum == 0:
            raise ValueError(
                "At least one output weight must be positive."
            )

        # Normalize without overflowing a sum of very large weights.
        weights = weights / maximum
        return weights / weights.sum()


@register_lag_selector("common_order_ic")
class CommonOrderICLagSelector(BaseCommonOrderLagSelector):
    """Select one common order using weighted per-output IC scores."""

    def __init__(
        self,
        max_lag: int = 12,
        *,
        use_lag_zero: bool = False,
        output_weights: Mapping[str, float] | None = None,
        **scoring_params: Any,
    ) -> None:
        _validate_scoring_params(
            scoring_params,
            {
                "use_bic",
                "complexity_measure",
                "variance_floor_rel",
                "score_atol",
                "score_rtol",
                "rcond",
            },
        )

        scorer = ICLagSelector(
            max_lag=max_lag,
            min_ar_order=0,
            prune_lags=False,
            **scoring_params,
        )

        super().__init__(
            max_lag=max_lag,
            use_lag_zero=use_lag_zero,
            output_weights=output_weights,
            scorer=scorer,
        )

    def _score_order(
        self,
        responses: np.ndarray,
        designs: tuple[np.ndarray, ...],
        weights: np.ndarray,
    ) -> float:
        total = 0.0

        for output_index, weight in enumerate(weights):
            if weight == 0:
                continue

            score = self._scorer._score(
                responses[:, output_index],
                designs[output_index],
            )
            total += float(weight) * score

        return float(total)


@register_lag_selector("common_order_cv")
class CommonOrderCVLagSelector(BaseCommonOrderLagSelector):
    """Select one common order using weighted time-series CV loss."""

    def __init__(
        self,
        max_lag: int = 12,
        *,
        use_lag_zero: bool = False,
        output_weights: Mapping[str, float] | None = None,
        normalize_outputs: bool = True,
        **scoring_params: Any,
    ) -> None:
        _validate_scoring_params(
            scoring_params,
            {
                "cv_folds",
                "cv_metric",
                "cv_gap",
                "cv_test_size",
                "cv_max_train_size",
                "score_atol",
                "score_rtol",
                "rcond",
            },
        )

        if not isinstance(normalize_outputs, bool):
            raise TypeError(
                "normalize_outputs must be a bool."
            )

        self.normalize_outputs = normalize_outputs

        scorer = CVLagSelector(
            max_lag=max_lag,
            min_ar_order=0,
            prune_lags=False,
            **scoring_params,
        )

        super().__init__(
            max_lag=max_lag,
            use_lag_zero=use_lag_zero,
            output_weights=output_weights,
            scorer=scorer,
        )

    def _score_order(
        self,
        responses: np.ndarray,
        designs: tuple[np.ndarray, ...],
        weights: np.ndarray,
    ) -> float:
        splits = self._scorer._make_cv_splits(
            len(responses)
        )

        total_loss = 0.0
        total_observations = 0

        for fold_number, (train_idx, val_idx) in enumerate(
            splits,
            start=1,
        ):
            fold_loss = 0.0

            for output_index, weight in enumerate(weights):
                if weight == 0:
                    continue

                y = responses[:, output_index]
                design = designs[output_index]

                try:
                    coefficients, _, _ = (
                        self._scorer._fit_auxiliary(
                            y[train_idx],
                            design[train_idx],
                        )
                    )

                    predictions = (
                        _augment_intercept(design[val_idx])
                        @ coefficients
                    )
                    residuals = y[val_idx] - predictions

                    scale = 1.0

                    if self.normalize_outputs:
                        scale = float(np.std(y[train_idx]))

                        if not np.isfinite(scale):
                            raise CandidateEvaluationError(
                                "Training response scale is non-finite."
                            )

                        if scale == 0:
                            scale = 1.0

                            diagnostics = (
                                self._scorer.diagnostics_
                            )
                            diagnostics[
                                "constant_training_scales"
                            ] = diagnostics.get(
                                "constant_training_scales", 0
                            ) + 1

                    residuals = residuals / scale

                    if not np.isfinite(residuals).all():
                        raise CandidateEvaluationError(
                            "Validation residuals are non-finite."
                        )

                    if self._scorer.cv_metric == "mse":
                        loss = float(residuals @ residuals)
                    else:
                        loss = float(
                            np.abs(residuals).sum()
                        )

                    if not np.isfinite(loss):
                        raise CandidateEvaluationError(
                            "Validation loss is non-finite."
                        )

                except (
                    CandidateEvaluationError,
                    np.linalg.LinAlgError,
                    FloatingPointError,
                    OverflowError,
                ) as exc:
                    raise CandidateEvaluationError(
                        f"Output index {output_index}, "
                        f"CV fold {fold_number}: {exc}"
                    ) from exc

                fold_loss += float(weight) * loss

            total_loss += fold_loss
            total_observations += len(val_idx)

        return float(
            total_loss / total_observations
        )