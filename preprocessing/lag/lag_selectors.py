import numpy as np
from numpy.linalg import lstsq
from numpy import log
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Optional, Sequence, Tuple
from ...core.lag_config import LagSelectionResult

import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR


class BaseLagSelector(ABC):
    """
    Abstract base class for lag selection in multivariate time series.

    This class defines the common interface and utilities used by
    concrete lag selectors, such as IC-based, CV-based, and VAR-based
    selectors.

    Parameters
    ----------
    max_lag : int
        Maximum lag to consider (upper bound for selected lags).
    center : bool, default True
        If True, subtract column-wise mean from X before fitting.
    use_lag_zero : bool, default False
        If True, the generated mask and column offsets assume that each
        predictor has an additional lag0 (current value) column in the
        lagged design matrix. Concrete selectors are responsible for
        deciding which lag0 entries are allowed (1) or forbidden (0)
        per (target, predictor) pair.
    target_indices : array-like of int or None, default None
        Indices of variables that should be treated as targets. If None,
        all variables are used as targets.
    """

    def __init__(self,
                 max_lag: int,
                 center: bool = True,
                 use_lag_zero: bool = False,
                 target_indices: Optional[Sequence[int]] = None):
        self.max_lag = max_lag
        self.center = center
        self.use_lag_zero = use_lag_zero
        self.target_indices = None if target_indices is None else list(target_indices)

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> LagSelectionResult:
        """
        Fit lag selection on input multivariate time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input multivariate time series, rows are time steps,
            columns are variables.

        Returns
        -------
        result : LagSelectionResult
            Lag selection result containing AR lags, per-pair lags,
            per-predictor maxima, column offsets, and mask.
        """
        X_proc = self._preprocess(X)
        T, D = X_proc.shape

        if self.target_indices is None:
            targets = list(range(D))
        else:
            targets = list(self.target_indices)

        ar_lags = np.zeros(D, dtype=int)
        pred_lag_matrix = np.zeros((D, D), dtype=int)

        # Delegate to concrete implementation for per-target lag selection
        self._select_lags(X_proc, targets, ar_lags, pred_lag_matrix)

        # Compute max_lags_per_pred and column offsets, then build mask
        max_lags_per_pred = pred_lag_matrix.max(axis=0)
        col_offsets, total_cols = self._compute_col_offsets(max_lags_per_pred)
        mask = self._build_mask(pred_lag_matrix, max_lags_per_pred, col_offsets)

        return LagSelectionResult(
            ar_lags=ar_lags,
            pred_lag_matrix=pred_lag_matrix,
            max_lags_per_pred=max_lags_per_pred,
            col_offsets=col_offsets,
            mask=mask,
        )

    # ------------------------------------------------------------------
    # ABSTRACT HOOK
    # ------------------------------------------------------------------
    @abstractmethod
    def _select_lags(self,
                     X: np.ndarray,
                     targets: Sequence[int],
                     ar_lags: np.ndarray,
                     pred_lag_matrix: np.ndarray) -> None:
        """
        Select lags for given targets and fill ar_lags and pred_lag_matrix.

        Concrete subclasses must implement this method.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Preprocessed input data.
        targets : sequence of int
            Indices of variables to be treated as targets.
        ar_lags : ndarray of shape (n_features,)
            Array to be filled with selected AR lags per variable.
        pred_lag_matrix : ndarray of shape (n_features, n_features)
            Matrix to be filled with maximum lags per (target, predictor) pair.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # SHARED UTILITIES
    # ------------------------------------------------------------------
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Center data column-wise if requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input time series data.

        Returns
        -------
        X_proc : ndarray of shape (n_samples, n_features)
            Preprocessed data.
        """
        X = np.asarray(X, dtype=float)
        if self.center:
            X = X - X.mean(axis=0, keepdims=True)
        return X

    def _compute_col_offsets(self,
                             max_lags_per_pred: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute starting column offsets for each predictor block.

        Parameters
        ----------
        max_lags_per_pred : ndarray of shape (n_features,)
            Maximum lag per predictor.

        Returns
        -------
        col_offsets : ndarray of shape (n_features,)
            Starting column index of each predictor block.
        total_cols : int
            Total number of columns in the lagged design matrix and mask.
        """
        max_lags_per_pred = np.asarray(max_lags_per_pred, dtype=int)
        n_features = max_lags_per_pred.shape[0]

        col_offsets = np.zeros(n_features, dtype=int)
        total_cols = 0
        for j in range(n_features):
            col_offsets[j] = total_cols
            n_cols_j = max_lags_per_pred[j]
            if self.use_lag_zero and n_cols_j > 0:
                n_cols_j += 1  # extra column for lag0
            total_cols += n_cols_j

        return col_offsets, total_cols

    def _build_mask(self,
                    pred_lag_matrix: np.ndarray,
                    max_lags_per_pred: np.ndarray,
                    col_offsets: np.ndarray) -> np.ndarray:
        """
        Build a binary mask for weights A based on selected lags.

        Interpretation:
        - pred_lag_matrix[i, j] = L > 0 means: allow lags 1..L of predictor j
          for target i.
        - pred_lag_matrix[i, j] = 0 means: predictor j is not used for target i.

        If use_lag_zero is True, an additional lag0 column is allocated per
        predictor j. The lag0 column is:
        - FORBIDDEN (0) for autoregression (i == j): the target cannot be
          on both sides of the equation without bias.
        - ALLOWED (1) for external predictors (i != j): the current unlagged
          value of the predictor can contribute to the target.

        Parameters
        ----------
        pred_lag_matrix : ndarray of shape (n_targets, n_features)
        max_lags_per_pred : ndarray of shape (n_features,)
        col_offsets : ndarray of shape (n_features,)

        Returns
        -------
        mask : ndarray of shape (n_targets, total_cols)
        """
        pred_lag_matrix = np.asarray(pred_lag_matrix, dtype=int)
        max_lags_per_pred = np.asarray(max_lags_per_pred, dtype=int)
        col_offsets = np.asarray(col_offsets, dtype=int)

        n_targets, n_features = pred_lag_matrix.shape
        total_cols = int(col_offsets[-1] + (
            (max_lags_per_pred[-1] + (1 if self.use_lag_zero and max_lags_per_pred[-1] > 0 else 0))
        )) if n_features > 0 else 0

        mask = np.zeros((n_targets, total_cols), dtype=int)

        for j in range(n_features):
            start_col = col_offsets[j]
            max_L_j = max_lags_per_pred[j]
            if max_L_j <= 0:
                continue

            # Optional lag0 column: allowed for external predictors,
            # but forbidden for autoregression (i == j).
            if self.use_lag_zero:
                col_lag0 = start_col
                # Allow lag0 only for non-autoregressive terms (external predictors)
                for i in range(n_targets):
                    if i != j:
                        # External predictor: lag0 is allowed
                        mask[i, col_lag0] = 1
                base = start_col + 1
            else:
                base = start_col

            for i in range(n_targets):
                L = pred_lag_matrix[i, j]
                if L <= 0:
                    continue
                L_eff = min(L, max_L_j)
                mask[i, base:base + L_eff] = 1

        return mask


# ======================================================================
# IC-BASED SELECTOR (AIC/BIC)
# ======================================================================

class ICLagSelector(BaseLagSelector):
    """
    Lag selector using information criteria (AIC or BIC) for ARX-style models.

    This selector:
    - for each target variable i, selects an autoregressive lag based on
      AIC or BIC,
    - for each predictor j, selects a maximum lag based on the same criterion,
    - optionally applies backward pruning in the multivariate model context
      (not included in this minimal sketch, but can be added as in your
      previous implementation),
    - returns AR lags, per-pair maximum lags, and the corresponding mask.

    Parameters
    ----------
    max_lag : int
        Maximum lag to consider (inclusive).
    center : bool, default True
        If True, subtract column-wise mean from X before fitting.
    use_lag_zero : bool, default False
        If True, lag0 columns are included in the mask/design; by default
        they are allowed for all predictors and targets (subclasses or
        callers can refine the mask, e.g. to forbid lag0 for i==j).
    use_bic : bool, default False
        If True, use BIC instead of AIC as the information criterion.
    n_jobs : int, default -1
        Number of parallel jobs over target variables. -1 uses all cores.
    target_indices : array-like of int or None, default None
        Indices of variables to treat as targets. If None, all variables
        are used as targets.
    """

    def __init__(self,
                 max_lag: int,
                 center: bool = True,
                 use_lag_zero: bool = False,
                 use_bic: bool = False,
                 n_jobs: int = -1,
                 target_indices: Optional[Sequence[int]] = None):
        super().__init__(max_lag=max_lag,
                         center=center,
                         use_lag_zero=use_lag_zero,
                         target_indices=target_indices)
        self.use_bic = use_bic
        self.n_jobs = n_jobs

    # -- scoring --------------------------------------------------------
    def _ic_score_linear(self, y: np.ndarray, X: np.ndarray) -> float:
        """
        Information-criterion score (AIC or BIC) for y ~ X with intercept.

        Lower scores are better.
        """
        n, p = X.shape
        if p == 0:
            mu = y.mean()
            resid = y - mu
            rss = float(np.sum(resid**2))
            k = 1
        else:
            X_aug = np.column_stack([np.ones(n), X])
            beta, _, _, _ = lstsq(X_aug, y, rcond=None)
            resid = y - X_aug @ beta
            rss = float(np.sum(resid**2))
            k = p + 1

        sigma2 = rss / n
        if sigma2 <= 0:
            sigma2 = 1e-12

        if self.use_bic:
            return k * log(n) + n * log(sigma2)
        else:
            return 2 * k + n * log(sigma2)

    def _design_matrix_arx(self,
                           X: np.ndarray,
                           target_idx: int,
                           target_lag: int,
                           pred_lag: int,
                           pred_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build design matrix for simple ARX model:

            y_t ~ y_{t - target_lag} + x_{t - pred_lag}
        """
        T, D = X.shape
        max_l = max(target_lag, pred_lag)

        if max_l == 0:
            y = X[:, target_idx]
            Phi = np.empty((len(y), 0), dtype=float)
            return y, Phi

        n_rows = T - max_l
        n_cols = (1 if target_lag > 0 else 0) + (1 if pred_lag > 0 else 0)
        Phi = np.empty((n_rows, n_cols), dtype=float)
        y = X[max_l:, target_idx]

        col = 0
        if target_lag > 0:
            Phi[:, col] = X[max_l - target_lag:T - target_lag, target_idx]
            col += 1
        if pred_lag > 0:
            Phi[:, col] = X[max_l - pred_lag:T - pred_lag, pred_idx]

        return y, Phi

    def _evaluate_target(self, X: np.ndarray, target_idx: int) -> Tuple[int, np.ndarray]:
        """
        Select AR lag and max lags per predictor for a single target.
        """
        T, D = X.shape

        # 1) AR lag selection
        best_ar_lag = 0
        best_ar_score = np.inf
        for lag in range(0, self.max_lag + 1):
            y, Phi = self._design_matrix_arx(X, target_idx, lag, 0, target_idx)
            score = self._ic_score_linear(y, Phi)
            if score < best_ar_score:
                best_ar_score = score
                best_ar_lag = lag

        # 2) predictor lags (max lag per predictor)
        best_pred_lags = np.zeros(D, dtype=int)
        for pred_idx in range(D):
            if pred_idx == target_idx:
                # AR handled separately
                continue

            best_lag = 0
            best_score = np.inf
            for lag in range(0, self.max_lag + 1):
                y, Phi = self._design_matrix_arx(
                    X, target_idx, best_ar_lag, lag, pred_idx
                )
                score = self._ic_score_linear(y, Phi)
                if score < best_score:
                    best_score = score
                    best_lag = lag

            best_pred_lags[pred_idx] = best_lag

        return best_ar_lag, best_pred_lags

    def _select_lags(self,
                     X: np.ndarray,
                     targets: Sequence[int],
                     ar_lags: np.ndarray,
                     pred_lag_matrix: np.ndarray) -> None:
        """
        IC-based lag selection implementation.
        """
        T, D = X.shape

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_target)(X, target_idx=i)
            for i in targets
        )

        for k, i in enumerate(targets):
            best_ar_lag, best_pred_lags = results[k]
            ar_lags[i] = best_ar_lag
            # store AR lag on diagonal as maximum lag for self-prediction
            best_pred_lags[i] = best_ar_lag
            pred_lag_matrix[i, :] = best_pred_lags


# ======================================================================
# CV-BASED SELECTOR
# ======================================================================

class CVLagSelector(ICLagSelector):
    """
    Lag selector using time-series cross-validation (blocked CV) for ARX models.

    This selector reuses the ARX structure of ICLagSelector, but replaces
    the information-criterion score with a time-series CV score (MSE or MAE).

    Parameters
    ----------
    max_lag : int
        Maximum lag to consider (inclusive).
    center : bool, default True
        If True, subtract column-wise mean from X before fitting.
    use_lag_zero : bool, default False
        If True, lag0 columns are included in the mask/design.
    cv_folds : int, default 5
        Number of folds for blocked time-series cross-validation.
    cv_metric : {'mse', 'mae'}, default 'mse'
        Error metric to minimize (MSE or MAE).
    n_jobs : int, default -1
        Number of parallel jobs over target variables. -1 uses all cores.
    target_indices : array-like of int or None, default None
        Indices of variables to treat as targets. If None, all variables
        are used as targets.
    """

    def __init__(self,
                 max_lag: int,
                 center: bool = True,
                 use_lag_zero: bool = False,
                 cv_folds: int = 5,
                 cv_metric: str = "mse",
                 n_jobs: int = -1,
                 target_indices: Optional[Sequence[int]] = None):
        super().__init__(max_lag=max_lag,
                         center=center,
                         use_lag_zero=use_lag_zero,
                         use_bic=False,
                         n_jobs=n_jobs,
                         target_indices=target_indices)
        self.cv_folds = cv_folds
        self.cv_metric = cv_metric

    def _cv_score_linear(self, y: np.ndarray, X: np.ndarray) -> float:
        """
        Blocked time-series cross-validation score for y ~ X with intercept.

        Lower scores are better. Uses simple forward-chaining blocked CV,
        averaging the validation error (MSE or MAE) across folds.
        """
        n, p = X.shape

        if p == 0:
            mu = y.mean()
            resid = y - mu
            if self.cv_metric == "mae":
                return float(np.mean(np.abs(resid)))
            else:
                return float(np.mean(resid**2))

        n_folds = max(2, int(self.cv_folds))
        fold_size = n // n_folds
        if fold_size < 1:
            # fallback to IC-based score if too few samples
            return self._ic_score_linear(y, X)

        errors = []
        for k in range(1, n_folds + 1):
            start_val = (k - 1) * fold_size
            end_val = n if k == n_folds else k * fold_size
            if start_val == 0:
                continue

            train_idx = np.arange(0, start_val)
            val_idx = np.arange(start_val, end_val)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            X_train_aug = np.column_stack([np.ones(len(train_idx)), X_train])
            X_val_aug = np.column_stack([np.ones(len(val_idx)), X_val])

            beta, _, _, _ = lstsq(X_train_aug, y_train, rcond=None)
            y_pred = X_val_aug @ beta
            resid = y_val - y_pred

            if self.cv_metric == "mae":
                err = float(np.mean(np.abs(resid)))
            else:
                err = float(np.mean(resid**2))
            errors.append(err)

        if not errors:
            return self._ic_score_linear(y, X)

        return float(np.mean(errors))

    def _evaluate_target(self, X: np.ndarray, target_idx: int) -> Tuple[int, np.ndarray]:
        """
        Select AR lag and max lags per predictor for a single target
        using CV-based score instead of IC.
        """
        T, D = X.shape

        # 1) AR lag selection
        best_ar_lag = 0
        best_ar_score = np.inf
        for lag in range(0, self.max_lag + 1):
            y, Phi = self._design_matrix_arx(X, target_idx, lag, 0, target_idx)
            score = self._cv_score_linear(y, Phi)
            if score < best_ar_score:
                best_ar_score = score
                best_ar_lag = lag

        # 2) predictor lags
        best_pred_lags = np.zeros(D, dtype=int)
        for pred_idx in range(D):
            if pred_idx == target_idx:
                continue

            best_lag = 0
            best_score = np.inf
            for lag in range(0, self.max_lag + 1):
                y, Phi = self._design_matrix_arx(
                    X, target_idx, best_ar_lag, lag, pred_idx
                )
                score = self._cv_score_linear(y, Phi)
                if score < best_score:
                    best_score = score
                    best_lag = lag

            best_pred_lags[pred_idx] = best_lag

        return best_ar_lag, best_pred_lags


# ======================================================================
# VAR-BASED SELECTOR
# ======================================================================

class VARLagSelector(BaseLagSelector):
    """
    Lag selector based on a VAR model with common lag order.

    This selector:
    - fits VAR models for lags 1..max_lag (or min_lag..max_lag) using
      information criteria (AIC, BIC, HQIC, FPE),
    - selects a single common lag order p for all variables,
    - sets ar_lags[i] = p and pred_lag_matrix[i, j] = p for all i, j,
    - builds mask consistent with use_lag_zero.

    Parameters
    ----------
    max_lag : int
        Maximum lag to evaluate in VAR.
    center : bool, default True
        If True, subtract column-wise mean from X before fitting VAR.
    use_lag_zero : bool, default False
        If True, lag0 columns are included in the mask/design; typically
        for VAR-based selectors you may want to allow lag0 for external
        regressors only, but that refinement can be handled by post-
        processing the mask.
    min_lag : int, default 1
        Minimum lag to evaluate in VAR.
    maximum : bool, default True
        If True, return the maximum optimal lag across criteria; if False,
        return the minimum (see auto_select_lag logic).
    target_indices : array-like of int or None, default None
        If given, only these variables are treated as targets when building
        pred_lag_matrix; by default, all variables are both endogenous and
        targets.
    """

    def __init__(self,
                 max_lag: int,
                 center: bool = True,
                 use_lag_zero: bool = False,
                 min_lag: int = 1,
                 maximum: bool = True,
                 target_indices: Optional[Sequence[int]] = None):
        super().__init__(max_lag=max_lag,
                         center=center,
                         use_lag_zero=use_lag_zero,
                         target_indices=target_indices)
        self.min_lag = min_lag
        self.maximum = maximum

    def _auto_select_lag(self, data: pd.DataFrame) -> int:
        """
        Determine optimal VAR lag order based on AIC, BIC, HQIC, and FPE.

        This is equivalent to the provided auto_select_lag function.
        """
        aic, bic, fpe, hqic = [], [], [], []
        model = VAR(data)
        p_grid = np.arange(max(0, self.min_lag), self.max_lag + 1)
        for p in p_grid:
            try:
                result = model.fit(p)
                aic.append(result.aic)
                bic.append(result.bic)
                fpe.append(result.fpe)
                hqic.append(result.hqic)
            except Exception:
                aic.append(np.inf)
                bic.append(np.inf)
                fpe.append(np.inf)
                hqic.append(np.inf)

        lags_metrics_df = pd.DataFrame(
            {'AIC': aic, 'BIC': bic, 'HQIC': hqic, 'FPE': fpe},
            index=p_grid
        )

        if self.maximum:
            return int(max(lags_metrics_df.idxmin(axis=0)))
        else:
            return int(min(lags_metrics_df.idxmin(axis=0)))

    def _select_lags(self,
                     X: np.ndarray,
                     targets: Sequence[int],
                     ar_lags: np.ndarray,
                     pred_lag_matrix: np.ndarray) -> None:
        """
        VAR-based lag selection implementation.

        Selects a single lag order p and applies it uniformly to all
        (target, predictor) pairs.
        """
        # VAR expects pandas DataFrame
        df = pd.DataFrame(X)
        p_opt = self._auto_select_lag(df)

        T, D = X.shape

        # Set AR lags and pairwise max lags
        for i in range(D):
            ar_lags[i] = p_opt

        for i in range(D):
            for j in range(D):
                pred_lag_matrix[i, j] = p_opt
