"""Shared heuristic lag selection and numerical fitting utilities."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ....core.lag_config import LagSpec
from ..registry import register_lag_selector
from .base import BaseLagSelector
from .splitting import time_series_splits
from .base import BaseLagSelector
from .utils import (
    _augment_intercept, 
    _nonnegative_finite, 
    _integer,
)

Reference = Literal["initial", "current"]


class CandidateEvaluationError(RuntimeError):
    """A particular candidate cannot be evaluated numerically."""



class BaseIncrementalLagSelector(BaseLagSelector):
    """Cumulative lag expansion followed by optional tail pruning.

    Numerical rank deficiency is allowed. Concrete scoring methods decide
    whether a fitted candidate is suitable for their criterion.

    Candidate searches are sequential. Dataset-level parallelism belongs
    to LagEngine, which supplies independent selector instances.
    """

    def __init__(
        self,
        max_lag: int = 12,
        *,
        use_lag_zero: bool = False,
        min_ar_order: int = 1,
        prune_lags: bool = True,
        addition_reference: Reference = "initial",
        pruning_reference: Reference = "current",
        score_atol: float = 1e-12,
        score_rtol: float = 1e-9,
        rcond: float | None = None,
    ) -> None:
        self.max_lag = _integer(
            max_lag, "max_lag", minimum=0
        )
        self.min_ar_order = _integer(
            min_ar_order, "min_ar_order", minimum=0
        )

        if self.min_ar_order > self.max_lag:
            raise ValueError(
                "min_ar_order cannot exceed max_lag."
            )

        for name, value in (
            ("addition_reference", addition_reference),
            ("pruning_reference", pruning_reference),
        ):
            if value not in {"initial", "current"}:
                raise ValueError(
                    f"{name} must be 'initial' or 'current'."
                )

        self.use_lag_zero = bool(use_lag_zero)
        self.prune_lags = bool(prune_lags)
        self.addition_reference = addition_reference
        self.pruning_reference = pruning_reference

        self.score_atol = _nonnegative_finite(
            score_atol, "score_atol"
        )
        self.score_rtol = _nonnegative_finite(
            score_rtol, "score_rtol"
        )
        self.rcond = (
            None
            if rcond is None
            else _nonnegative_finite(rcond, "rcond")
        )

    @abstractmethod
    def _score(
        self,
        y: np.ndarray,
        design: np.ndarray,
    ) -> float:
        """Return a finite lower-is-better score.

        Expected candidate-specific numerical failures must raise
        CandidateEvaluationError. Configuration/programming errors
        must not be converted to candidate failures.
        """
        raise NotImplementedError

    @abstractmethod
    def _accept_addition(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _accept_removal(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        raise NotImplementedError

    def fit(
        self,
        data: pd.DataFrame,
        *,
        outputs: Sequence[str] | None = None,
    ) -> LagSpec:
        names, targets, values = self._validate_input(
            data, outputs
        )

        if len(values) <= self.max_lag:
            raise ValueError(
                "The dataset must contain more rows than max_lag."
            )

        # Clear the previous fitted result before starting a new run.
        for name in (
            "lag_spec_",
            "feature_names_in_",
            "output_names_",
            "selection_start_",
            "n_selection_samples_",
        ):
            self.__dict__.pop(name, None)

        self.diagnostics_ = {
            "rank_deficient_fits": 0,
            "variance_floor_uses": 0,
            "failed_candidates": [],
        }

        start = self.max_lag
        response = values[start:]
        name_to_index = {
            name: index for index, name in enumerate(names)
        }

        first_lag = 0 if self.use_lag_zero else 1

        columns = {
            (name, lag): values[
                start - lag : len(values) - lag,
                index,
            ]
            for index, name in enumerate(names)
            for lag in range(first_lag, self.max_lag + 1)
        }

        result: LagSpec = {}

        for target in targets:
            y = response[:, name_to_index[target]]

            selected = self._initial_selection(
                y=y,
                target=target,
                predictors=names,
                columns=columns,
            )

            if self.prune_lags:
                selected = self._prune(
                    y=y,
                    target=target,
                    selected=selected,
                    columns=columns,
                )
            else:
                score = self._evaluate(
                    y,
                    selected,
                    columns,
                    target=target,
                    stage="combined",
                )

                if score is None:
                    raise ValueError(
                        f"The combined candidate for {target!r} "
                        "cannot be scored. See diagnostics_."
                    )

            result[target] = selected

        self.feature_names_in_ = names
        self.output_names_ = targets
        self.selection_start_ = start
        self.n_selection_samples_ = len(response)
        self.lag_spec_ = result

        return result

    def _initial_selection(
        self,
        *,
        y: np.ndarray,
        target: str,
        predictors: tuple[str, ...],
        columns: dict[tuple[str, int], np.ndarray],
    ) -> dict[str, tuple[int, ...]]:
        baseline = {name: () for name in predictors}
        baseline[target] = tuple(
            range(1, self.min_ar_order + 1)
        )

        ar_selected = self._expand_predictor(
            y=y,
            target=target,
            predictor=target,
            baseline=baseline,
            candidates=(
                tuple(range(1, order + 1))
                for order in range(
                    self.min_ar_order + 1,
                    self.max_lag + 1,
                )
            ),
            columns=columns,
        )

        selected = ar_selected.copy()
        first_lag = 0 if self.use_lag_zero else 1

        for predictor in predictors:
            if predictor == target:
                continue

            # Independent screening conditional on the same AR model.
            predictor_selected = self._expand_predictor(
                y=y,
                target=target,
                predictor=predictor,
                baseline=ar_selected,
                candidates=(
                    tuple(range(first_lag, last_lag + 1))
                    for last_lag in range(
                        first_lag,
                        self.max_lag + 1,
                    )
                ),
                columns=columns,
            )

            selected[predictor] = predictor_selected[predictor]

        return selected

    def _expand_predictor(
        self,
        *,
        y: np.ndarray,
        target: str,
        predictor: str,
        baseline: dict[str, tuple[int, ...]],
        candidates: Iterable[tuple[int, ...]],
        columns: dict[tuple[str, int], np.ndarray],
    ) -> dict[str, tuple[int, ...]]:
        best = baseline.copy()

        best_score = self._evaluate(
            y,
            best,
            columns,
            target=target,
            stage="addition_baseline",
        )
        initial_score = best_score

        for lags in candidates:
            candidate = baseline.copy()
            candidate[predictor] = lags

            score = self._evaluate(
                y,
                candidate,
                columns,
                target=target,
                stage="addition",
            )

            if score is None:
                continue

            # If the baseline was unscorable, the first valid candidate
            # becomes the starting point for subsequent comparisons.
            if best_score is None:
                best = candidate
                best_score = score
                initial_score = score
                continue

            reference_score = (
                initial_score
                if self.addition_reference == "initial"
                else best_score
            )
            assert reference_score is not None

            if (
                self._strictly_better(score, best_score)
                and self._accept_addition(
                    score, reference_score
                )
            ):
                best = candidate
                best_score = score

        if best_score is None:
            raise ValueError(
                f"No scorable candidate for output {target!r}, "
                f"predictor {predictor!r}. See diagnostics_."
            )

        return best

    def _prune(
        self,
        *,
        y: np.ndarray,
        target: str,
        selected: dict[str, tuple[int, ...]],
        columns: dict[tuple[str, int], np.ndarray],
    ) -> dict[str, tuple[int, ...]]:
        current = selected.copy()

        current_score = self._evaluate(
            y,
            current,
            columns,
            target=target,
            stage="pruning_baseline",
        )
        initial_score = current_score

        while True:
            reference_score = (
                initial_score
                if self.pruning_reference == "initial"
                else current_score
            )

            best_candidate = None
            best_score = None

            for predictor, lags in current.items():
                minimum_count = (
                    self.min_ar_order
                    if predictor == target
                    else 0
                )

                if len(lags) <= minimum_count:
                    continue

                candidate = current.copy()
                candidate[predictor] = lags[:-1]

                score = self._evaluate(
                    y,
                    candidate,
                    columns,
                    target=target,
                    stage="pruning",
                )

                if score is None:
                    continue

                # When the combined model was unscorable, allow
                # recovery through a valid one-step reduction.
                if (
                    reference_score is not None
                    and not self._accept_removal(
                        score, reference_score
                    )
                ):
                    continue

                if (
                    best_score is None
                    or self._strictly_better(score, best_score)
                ):
                    best_candidate = candidate
                    best_score = score

            if best_candidate is None:
                if current_score is None:
                    raise ValueError(
                        f"The combined model for {target!r} cannot "
                        "be scored, and no valid one-step pruning "
                        "candidate was found. See diagnostics_."
                    )
                return current

            current = best_candidate
            current_score = best_score

            if initial_score is None:
                initial_score = current_score

    def _evaluate(
        self,
        y: np.ndarray,
        lags: dict[str, tuple[int, ...]],
        columns: dict[tuple[str, int], np.ndarray],
        *,
        target: str,
        stage: str,
    ) -> float | None:
        active_columns = [
            columns[(predictor, lag)]
            for predictor, selected_lags in lags.items()
            for lag in selected_lags
        ]

        design = (
            np.column_stack(active_columns)
            if active_columns
            else np.empty((len(y), 0), dtype=float)
        )

        try:
            with np.errstate(
                over="raise",
                invalid="raise",
                divide="raise",
            ):
                score = float(self._score(y, design))

            if not np.isfinite(score):
                raise CandidateEvaluationError(
                    "The scoring method returned a non-finite value."
                )

            return score

        except (
            CandidateEvaluationError,
            np.linalg.LinAlgError,
            FloatingPointError,
            OverflowError,
        ) as exc:
            self.diagnostics_["failed_candidates"].append({
                "output": target,
                "stage": stage,
                "lags": lags.copy(),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            return None

    def _fit_auxiliary(
        self,
        y: np.ndarray,
        design: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Fit minimum-norm OLS without requiring full column rank."""
        augmented = _augment_intercept(design)

        coefficients, _, rank, _ = np.linalg.lstsq(
            augmented,
            y,
            rcond=self.rcond,
        )

        rank = int(rank)
        residuals = y - augmented @ coefficients

        if not (
            np.isfinite(coefficients).all()
            and np.isfinite(residuals).all()
        ):
            raise CandidateEvaluationError(
                "OLS produced non-finite coefficients or residuals."
            )

        if rank < augmented.shape[1]:
            self.diagnostics_["rank_deficient_fits"] += 1

        return coefficients, residuals, rank

    def _score_tolerance(
        self,
        first: float,
        second: float,
    ) -> float:
        return (
            self.score_atol
            + self.score_rtol * max(abs(first), abs(second))
        )

    def _strictly_better(
        self,
        candidate: float,
        reference: float,
    ) -> bool:
        return (
            candidate < reference
            and reference - candidate
            > self._score_tolerance(candidate, reference)
        )

    def _not_worse_than(
        self,
        candidate: float,
        limit: float,
    ) -> bool:
        return (
            candidate <= limit
            or candidate - limit
            <= self._score_tolerance(candidate, limit)
        )

    @staticmethod
    def _validate_input(
        data: pd.DataFrame,
        outputs: Sequence[str] | None,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        np.ndarray,
    ]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError("data must not be empty.")

        if not data.columns.is_unique:
            raise ValueError(
                "data contains duplicated column names."
            )

        if not all(
            isinstance(name, str)
            for name in data.columns
        ):
            raise TypeError(
                "All column names must be strings."
            )

        names = tuple(data.columns)

        if isinstance(outputs, str):
            raise TypeError(
                "outputs must be a sequence, not a string."
            )

        targets = names if outputs is None else tuple(outputs)

        if not targets:
            raise ValueError(
                "At least one output must be requested."
            )

        if not all(
            isinstance(name, str)
            for name in targets
        ):
            raise TypeError(
                "Output names must be strings."
            )

        if len(targets) != len(set(targets)):
            raise ValueError(
                "outputs contains duplicated names."
            )

        missing = set(targets).difference(names)
        if missing:
            raise ValueError(
                f"Unknown outputs: {sorted(missing)!r}."
            )

        values = data.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "data must contain only finite numeric values."
            )

        return names, targets, values


@register_lag_selector("ic")
class ICLagSelector(BaseIncrementalLagSelector):
    """Cumulative lag selection using an AIC/BIC-based score.

    complexity_measure="columns" penalizes the representation size.
    With rank-deficient designs, this is a deliberate heuristic penalty.

    complexity_measure="rank" penalizes the dimension of the fitted
    linear prediction space.
    """

    def __init__(
        self,
        max_lag: int = 12,
        *,
        use_bic: bool = False,
        use_lag_zero: bool = False,
        min_ar_order: int = 1,
        delta_min_ic: float | None = 2.0,
        prune_lags: bool = True,
        delta_prune_ic: float = 2.0,
        addition_reference: Reference = "initial",
        pruning_reference: Reference = "current",
        complexity_measure: str = "columns",
        variance_floor_rel: float = 1e-12,
        score_atol: float = 1e-12,
        score_rtol: float = 1e-9,
        rcond: float | None = None,
    ) -> None:
        super().__init__(
            max_lag=max_lag,
            use_lag_zero=use_lag_zero,
            min_ar_order=min_ar_order,
            prune_lags=prune_lags,
            addition_reference=addition_reference,
            pruning_reference=pruning_reference,
            score_atol=score_atol,
            score_rtol=score_rtol,
            rcond=rcond,
        )

        if complexity_measure not in {"columns", "rank"}:
            raise ValueError(
                "complexity_measure must be 'columns' or 'rank'."
            )

        self.use_bic = bool(use_bic)
        self.complexity_measure = complexity_measure

        self.delta_min_ic = (
            None
            if delta_min_ic is None
            else _nonnegative_finite(
                delta_min_ic, "delta_min_ic"
            )
        )
        self.delta_prune_ic = _nonnegative_finite(
            delta_prune_ic, "delta_prune_ic"
        )

        self.variance_floor_rel = _nonnegative_finite(
            variance_floor_rel, "variance_floor_rel"
        )
        if self.variance_floor_rel == 0:
            raise ValueError(
                "variance_floor_rel must be positive."
            )

    def _score(
        self,
        y: np.ndarray,
        design: np.ndarray,
    ) -> float:
        _, residuals, rank = self._fit_auxiliary(y, design)

        n = len(y)

        # Numerical rank deficiency is allowed, but a saturated
        # auxiliary model is not used for this IC score.
        if n <= rank:
            raise CandidateEvaluationError(
                "IC candidate has no residual degrees of freedom: "
                f"n={n}, rank={rank}."
            )

        k = (
            design.shape[1] + 1
            if self.complexity_measure == "columns"
            else rank
        )

        rss = float(residuals @ residuals)

        if not np.isfinite(rss):
            raise CandidateEvaluationError(
                "Residual sum of squares is non-finite."
            )

        # y is identical across candidates for a given output,
        # so this reference scale does not depend on the candidate.
        reference_variance = float(np.var(y))

        variance_floor = max(
            self.variance_floor_rel * reference_variance,
            np.finfo(float).tiny,
        )

        log_floor = float(np.log(variance_floor))

        if rss == 0:
            log_variance = log_floor
            floor_used = True
        else:
            raw_log_variance = float(
                np.log(rss) - np.log(n)
            )
            log_variance = max(
                raw_log_variance,
                log_floor,
            )
            floor_used = raw_log_variance < log_floor

        if floor_used:
            self.diagnostics_["variance_floor_uses"] += 1

        penalty = (
            k * np.log(n)
            if self.use_bic
            else 2.0 * k
        )

        return float(n * log_variance + penalty)

    def _accept_addition(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        delta = (
            0.0
            if self.delta_min_ic is None
            else self.delta_min_ic
        )

        return self._strictly_better(
            candidate_score,
            reference_score - delta,
        )

    def _accept_removal(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        return self._not_worse_than(
            candidate_score,
            reference_score + self.delta_prune_ic,
        )


@register_lag_selector("cv")
class CVLagSelector(BaseIncrementalLagSelector):
    """Cumulative lag selection using time-series validation loss.

    Candidate fitting uses minimum-norm OLS. Rank deficiency and
    underdetermined training regressions are permitted.
    """

    def __init__(
        self,
        max_lag: int = 12,
        *,
        use_lag_zero: bool = False,
        min_ar_order: int = 1,
        cv_folds: int = 5,
        cv_metric: str = "mse",
        cv_gap: int = 0,
        cv_test_size: int | None = None,
        cv_max_train_size: int | None = None,
        delta_min_rel_cv: float | None = None,
        prune_lags: bool = False,
        delta_prune_rel_cv: float = 0.02,
        addition_reference: Reference = "initial",
        pruning_reference: Reference = "current",
        score_atol: float = 1e-12,
        score_rtol: float = 1e-9,
        rcond: float | None = None,
    ) -> None:
        super().__init__(
            max_lag=max_lag,
            use_lag_zero=use_lag_zero,
            min_ar_order=min_ar_order,
            prune_lags=prune_lags,
            addition_reference=addition_reference,
            pruning_reference=pruning_reference,
            score_atol=score_atol,
            score_rtol=score_rtol,
            rcond=rcond,
        )

        self.cv_folds = _integer(
            cv_folds, "cv_folds", minimum=2
        )
        self.cv_gap = _integer(
            cv_gap, "cv_gap", minimum=0
        )

        if cv_metric not in {"mse", "mae"}:
            raise ValueError(
                "cv_metric must be 'mse' or 'mae'."
            )
        self.cv_metric = cv_metric

        self.cv_test_size = (
            None
            if cv_test_size is None
            else _integer(
                cv_test_size,
                "cv_test_size",
                minimum=1,
            )
        )
        self.cv_max_train_size = (
            None
            if cv_max_train_size is None
            else _integer(
                cv_max_train_size,
                "cv_max_train_size",
                minimum=1,
            )
        )

        self.delta_min_rel_cv = (
            None
            if delta_min_rel_cv is None
            else _nonnegative_finite(
                delta_min_rel_cv,
                "delta_min_rel_cv",
            )
        )

        if (
            self.delta_min_rel_cv is not None
            and self.delta_min_rel_cv >= 1
        ):
            raise ValueError(
                "delta_min_rel_cv must be less than 1."
            )

        self.delta_prune_rel_cv = _nonnegative_finite(
            delta_prune_rel_cv,
            "delta_prune_rel_cv",
        )

    def _make_cv_splits(
        self,
        n_samples: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        return time_series_splits(
            n_samples,
            n_splits=self.cv_folds,
            test_size=self.cv_test_size,
            gap=self.cv_gap,
            max_train_size=self.cv_max_train_size,
        )

    def _score(
        self,
        y: np.ndarray,
        design: np.ndarray,
    ) -> float:
        # Split-configuration errors propagate as ValueError.
        # They are not silently treated as candidate failures.
        splits = self._make_cv_splits(len(y))

        total_loss = 0.0
        total_observations = 0

        for fold_number, (train_idx, val_idx) in enumerate(
            splits,
            start=1,
        ):
            try:
                coefficients, _, _ = self._fit_auxiliary(
                    y[train_idx],
                    design[train_idx],
                )

                predictions = (
                    _augment_intercept(design[val_idx])
                    @ coefficients
                )

                residuals = y[val_idx] - predictions

                if not np.isfinite(residuals).all():
                    raise CandidateEvaluationError(
                        "Validation residuals are non-finite."
                    )

                if self.cv_metric == "mse":
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
                    f"CV fold {fold_number} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            total_loss += loss
            total_observations += len(val_idx)

        return float(
            total_loss / total_observations
        )

    def _accept_addition(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        relative_delta = (
            0.0
            if self.delta_min_rel_cv is None
            else self.delta_min_rel_cv
        )

        limit = reference_score * (
            1.0 - relative_delta
        )

        return self._strictly_better(
            candidate_score,
            limit,
        )

    def _accept_removal(
        self,
        candidate_score: float,
        reference_score: float,
    ) -> bool:
        limit = reference_score * (
            1.0 + self.delta_prune_rel_cv
        )

        return self._not_worse_than(
            candidate_score,
            limit,
        )