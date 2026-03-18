import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from numpy.linalg import lstsq
from math import log
from joblib import Parallel, delayed


def auto_select_lag(data: pd.DataFrame,  min_lag: int = 1 , max_lag: int = 20,  maximum: bool = True) -> int:
    """
    Determines the optimal lag order for a VAR model using information criteria.

    Parameters
    --------------
    data : pd.DataFrame
        Multivariate time series data for VAR modeling.
    max_lag : int, optional
        Maximum lag order to evaluate (default=20).
    maximum : bool, optional
        If True, returns the maximum optimal lag across criteria; 
        if False, returns the minimum (default=True).

    Return
    --------------
    int
        Optimal lag order based on selected criteria (AIC, BIC, HQIC, FPE).

    Notes:
    ------
    - Fits VAR models for lags 1 through max_lag.
    - Computes AIC, BIC, FPE, and HQIC for each lag.
    - Selects the lag that minimizes each criterion, then returns either the max or min of these lags.
    """

    
    aic, bic, fpe, hqic = [], [], [], []
    model = VAR(data) 
    p = np.arange(max(0,min_lag),max_lag)
    for i in p:
        try:
            result = model.fit(i)
            aic.append(result.aic)
            bic.append(result.bic)
            fpe.append(result.fpe)
            hqic.append(result.hqic)
        except:
            aic.append(np.inf)
            bic.append(np.inf)
            fpe.append(np.inf)
            hqic.append(np.inf)
    lags_metrics_df = pd.DataFrame({'AIC': aic, 
                                    'BIC': bic, 
                                    'HQIC': hqic,
                                    'FPE': fpe}, 
                                   index=p) 
    if maximum:
        return max(lags_metrics_df.idxmin(axis=0))
    else:
        return min(lags_metrics_df.idxmin(axis=0))
    
def create_lagged_data(data: pd.DataFrame, min_lags: np.ndarray, max_lags: np.ndarray) -> np.ndarray:
    """
    Generates lagged features for multivariate time-series data.

    Each column of ``data`` is expanded into a sequence of delayed versions.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data (columns = variables).
    min_lags : np.ndarray
        Minimum lag to include for each variable (shape = n_columns).
    max_lags : np.ndarray
        Maximum lag to include for each variable (shape = n_columns).

    Returns
    -------
    np.ndarray
        Lagged feature matrix with shape ``(n_samples - max(max_lags), sum(max_lags -
        min_lags + 1))``. Rows corresponding to insufficient history are removed.

    Notes
    -----
    - The caller is responsible for aligning target values (e.g. dropping the first
      ``max(max_lags)`` rows of the original data).
    - Any NaNs produced by differencing or shifting are removed by slicing; the
      returned array should not contain NaNs.
    """
    # convert to numpy arrays and validate
    x = data.values.copy()
    min_lags = np.asarray(min_lags, dtype=int)
    max_lags = np.asarray(max_lags, dtype=int)
    if min_lags.shape != max_lags.shape:
        raise ValueError("min_lags and max_lags must have the same shape")
    if np.any(min_lags < 0) or np.any(max_lags < 0):
        raise ValueError("lag values must be non-negative")
    if np.any(min_lags > max_lags):
        raise ValueError("each min_lag must be <= corresponding max_lag")

    n_vars = x.shape[1]
    total_cols = (max_lags - min_lags + 1).sum(dtype=int)
    X = np.zeros((x.shape[0], total_cols), dtype=x.dtype)

    data_indexes = (max_lags - min_lags + 1).cumsum(dtype=int)
    data_indexes = np.concatenate([[0], data_indexes], dtype=int)

    for k in range(n_vars):
        lagged_list = []
        for i in range(int(min_lags[k]), int(max_lags[k] + 1)):
            # roll moves data down by i rows; pad front with nan
            col = np.roll(x[:, [k]], i, axis=0)
            if i > 0:
                col[:i, 0] = np.nan
            lagged_list.append(col)
        X[:, data_indexes[k] : data_indexes[k + 1]] = np.concatenate(lagged_list, axis=1)

    # drop rows that contain NaNs (first max_lags.max() rows)
    valid_start = int(max_lags.max())
    X = X[valid_start:, :]
    return X

def make_static(data, order: int):
    """Apply differencing to a univariate series to achieve stationarity.

    The original implementation used ``np.roll`` incorrectly which produced
    zero arrays and silently ignored the requested order.  This version
    leverages pandas for clarity and correct handling of NaNs.

    Parameters
    ----------
    data : array-like or pd.Series
        Input time series.
    order : int
        Number of differences to apply.  ``order <= 0`` returns the input
        unchanged.

    Returns
    -------
    np.ndarray
        Differenced series with ``order`` NaNs at the beginning.
    """
    if order <= 0:
        return np.asarray(data)
    s = pd.Series(data).astype(float)
    for _ in range(order):
        s = s.diff()
    return s.values

class ARXLagSelector:
    """
    Select AR and cross-variable lags for a multivariate time series
    using information criteria (AIC/BIC) or time-series cross-validation,
    with optional backward pruning of redundant lags.

    Parameters
    ----------
    max_lag : int
        Maximum lag to consider (lags 1..max_lag).
    center : bool, default True
        If True, subtract column-wise mean from X before fitting.
    n_jobs : int, default -1
        Number of parallel jobs over target variables. -1 uses all cores.
    score_mode : {'ic', 'cv'}, default 'ic'
        'ic'  -> use information criterion (AIC/BIC).
        'cv'  -> use time-series cross-validation.
    use_bic : bool, default False
        If True (and score_mode == 'ic'), use BIC instead of AIC.
    cv_folds : int, default 5
        Number of folds for time-series cross-validation (when score_mode == 'cv').
    cv_metric : {'mse', 'mae'}, default 'mse'
        Error metric to minimize in CV.
    prune_lags : bool, default False
        If True, run an additional backward pruning phase on selected lags.
    delta_prune_ic : float, default 2.0
        Maximum allowed worsening of IC score (AIC/BIC) when removing a lag.
        Only used when score_mode == 'ic'.
    delta_prune_rel_cv : float, default 0.02
        Maximum allowed relative increase of CV error when removing a lag,
        e.g. 0.02 means up to 2% worse CV error is allowed.
        Only used when score_mode == 'cv'.
    """

    def __init__(self,
                 max_lag: int,
                 target_indices=None,
                 center: bool = True,
                 n_jobs: int = -1,
                 score_mode: str = "ic",
                 use_bic: bool = False,
                 cv_folds: int = 3,
                 cv_metric: str = "mse",
                 prune_lags: bool = False,
                 delta_prune_ic: float = 2.0,
                 delta_prune_rel_cv: float = 0.02,
                 delta_min_ic: float = 2.0,
                 delta_min_rel_cv: float = 0.02):
        self.max_lag = max_lag
        self.center = center
        self.n_jobs = n_jobs
        
        self.score_mode = score_mode
        self.use_bic = use_bic
        
        self.cv_folds = cv_folds
        self.cv_metric = cv_metric
        
        self.prune_lags = prune_lags
        self.delta_prune_ic = delta_prune_ic
        self.delta_prune_rel_cv = delta_prune_rel_cv

        self.delta_min_ic = delta_min_ic
        self.delta_min_rel_cv = delta_min_rel_cv
      
        self.target_indices = target_indices 

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
    
    def _min_improvement(self, score: float, best_score: float) -> bool:
        if self.score_mode == "ic":
            return score < best_score - self.delta_min_ic
        elif self.score_mode == "cv":
            return score < best_score * (1 - self.delta_min_rel_cv)
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode!r}")

    def _ic_score_linear(self, y: np.ndarray, X: np.ndarray) -> float:
        """
        Information-criterion score (AIC or BIC) for y ~ X with intercept.
        Lower is better.
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
            return k * log(n) + n * log(sigma2)  # BIC
        else:
            return 2 * k + n * log(sigma2)       # AIC

    def _cv_score_linear(self, y: np.ndarray, X: np.ndarray) -> float:
        """
        Time-series cross-validation score for y ~ X with intercept.

        Implements a simple blocked time-series CV without external libraries:
        splits the series into `cv_folds` consecutive blocks, and for each fold
        uses all previous data as training and the current block as validation.

        Returns the average validation error (MSE or MAE). Lower is better.
        """
        n, p = X.shape

        # Intercept-only model
        if p == 0:
            mu = y.mean()
            resid = y - mu
            if self.cv_metric == "mae":
                return float(np.mean(np.abs(resid)))
            else:
                return float(np.mean(resid**2))

        # Ensure at least 2 folds and that folds are not empty
        n_folds = max(2, int(self.cv_folds))
        fold_size = n // n_folds
        if fold_size < 1:
            # Too few samples for requested folds: fall back to one-shot fit
            return self._ic_score_linear(y, X)

        errors = []

        # For fold k, use indices [0 : k*fold_size) as train,
        # and [k*fold_size : (k+1)*fold_size) as validation.
        # Last fold may include remaining samples.
        for k in range(1, n_folds + 1):
            start_val = (k - 1) * fold_size
            if k < n_folds:
                end_val = k * fold_size
            else:
                end_val = n  # include the rest in the last fold

            # Require non-empty training set
            if start_val == 0:
                continue

            train_idx = np.arange(0, start_val)
            val_idx = np.arange(start_val, end_val)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit linear model with intercept
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
            # Fallback if something went wrong with splitting
            return self._ic_score_linear(y, X)

        return float(np.mean(errors))

    def _score_linear(self, y: np.ndarray, X: np.ndarray) -> float:
        """
        Unified scoring function that dispatches to IC or CV backend.
        """
        if self.score_mode == "ic":
            return self._ic_score_linear(y, X)
        elif self.score_mode == "cv":
            return self._cv_score_linear(y, X)
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode!r}")

    def _design_matrix_arx(
        self,
        X: np.ndarray,
        target_idx: int,
        target_lag: int,
        pred_lag: int,
        pred_idx: int
    ):
        """
        Build design matrix for a simple ARX model:

            y_t ~ y_{t - target_lag} + x_{t - pred_lag}

        Both lags can be zero, which means excluding that term.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input time series (columns are variables).
        target_idx : int
            Index of the target variable.
        target_lag : int
            Lag for the autoregressive term (0 means no AR term).
        pred_lag : int
            Lag for the predictor variable (0 means no predictor).
        pred_idx : int
            Index of the predictor variable.

        Returns
        -------
        y : ndarray of shape (n_effective_samples,)
            Target values aligned with the lagged predictors.
        Phi : ndarray of shape (n_effective_samples, n_features_in_model)
            Design matrix consisting of the selected lagged terms.
        """
        T, D = X.shape
        max_l = max(target_lag, pred_lag)

        # If both lags are zero, we only use an intercept (no features)
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

    def _evaluate_pair_for_target(self, X: np.ndarray, target_idx: int):
        """
        Select best AR lag for the target and best predictor lag per variable.

        For a given target variable i:
        - first find the best AR lag (0..max_lag) using the information criterion,
        - then, for each predictor j, find the best lag (0..max_lag) when AR lag is fixed.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Preprocessed time series data.
        target_idx : int
            Index of the target variable.

        Returns
        -------
        best_ar_lag : int
            Selected AR lag for the target variable (0 means no AR term).
        best_pred_lags : ndarray of shape (n_features,)
            For each predictor j, best lag (0..max_lag) for predicting the target.
            For j == target_idx, this is always 0 (AR lag is handled separately).
        """
        T, D = X.shape

        # 1) Select best AR lag for the target (without predictors)
        best_ar_lag = 0
        best_ar_score = np.inf

        for lag in range(0, self.max_lag + 1):  # 0 = no AR term
            y, Phi = self._design_matrix_arx(X, target_idx, lag, 0, target_idx)
            score = self._score_linear(y, Phi)
            if score < best_ar_score:
                best_ar_score = score
                best_ar_lag = lag

        # 2) For each predictor, select the best lag given the AR lag
        best_pred_lags = np.zeros(D, dtype=int)

        for pred_idx in range(D):
            if pred_idx == target_idx:
                # AR lag is stored separately in best_ar_lag
                best_pred_lags[pred_idx] = 0
                continue

            best_lag = 0
            best_score = best_ar_score

            for lag in range(0, self.max_lag + 1):  # 0 = exclude predictor
                y, Phi = self._design_matrix_arx(
                    X, target_idx, best_ar_lag, lag, pred_idx
                )
                score = self._score_linear(y, Phi)
                if self._min_improvement(score, best_score):
                    best_score = score
                    best_lag = lag

            best_pred_lags[pred_idx] = best_lag

        return best_ar_lag, best_pred_lags

    def _build_design_for_target(self, X: np.ndarray, target_idx: int,
                                 active_pairs: list[tuple[int, int]]):
        """
        Build design matrix y, Phi for a given target and set of (pred_idx, lag).
        """
        T, D = X.shape
        if not active_pairs:
            y = X[:, target_idx]
            Phi = np.empty((len(y), 0), dtype=float)
            return y, Phi

        max_l = max(l for _, l in active_pairs)
        n_rows = T - max_l
        y = X[max_l:, target_idx]
        Phi = np.empty((n_rows, len(active_pairs)), dtype=float)

        for k, (j, lag) in enumerate(active_pairs):
            Phi[:, k] = X[max_l - lag:T - lag, j]

        return y, Phi

    def _prune_lags_for_target(self,
                               X: np.ndarray,
                               target_idx: int,
                               ar_lag: int,
                               pred_lags: np.ndarray):
        """
        Backward pruning of lags for a single target.

        Parameters
        ----------
        X : ndarray (T, D)
        target_idx : int
        ar_lag : int
        pred_lags : ndarray (D,)
            Current lag per predictor (including diagonal, may be 0).
        base_score_ar : float
            Score of AR-only model for this target (from first phase).

        Returns
        -------
        new_ar_lag : int
        new_pred_lags : ndarray (D,)
        """
        T, D = X.shape

        # Build initial set of active (pred_idx, lag)
        active = []
        for j in range(D):
            lag = pred_lags[j]
            if lag > 0:
                active.append((j, lag))

        # If AR lag is not in pred_lags diagonal, add it explicitly
        if ar_lag > 0 and pred_lags[target_idx] == 0:
            active.append((target_idx, ar_lag))

        # If no active lags, nothing to prune
        if not active:
            return ar_lag, pred_lags

        # Initial full-model score
        y_full, Phi_full = self._build_design_for_target(X, target_idx, active)
        full_score = self._score_linear(y_full, Phi_full)

        # For IC: we'll compare absolute score differences
        # For CV: we'll compare relative changes
        improved = True
        while improved and len(active) > 0:
            improved = False
            best_candidate_score = full_score
            best_to_remove = None

            for idx_to_remove in range(len(active)):
                candidate = active[:idx_to_remove] + active[idx_to_remove+1:]
                y_c, Phi_c = self._build_design_for_target(X, target_idx, candidate)
                score_c = self._score_linear(y_c, Phi_c)

                if self.score_mode == "ic":
                    # Accept removal if score does not worsen by more than delta_prune_ic
                    if score_c <= full_score + self.delta_prune_ic:
                        # prefer simpler model with equal/better score
                        if best_to_remove is None or score_c < best_candidate_score:
                            best_candidate_score = score_c
                            best_to_remove = idx_to_remove
                elif self.score_mode == "cv":  # 'cv' mode
                    if score_c<=full_score*(1+self.delta_prune_rel_cv):
                        if best_to_remove is None or score_c < best_candidate_score:
                            best_candidate_score = score_c
                            best_to_remove = idx_to_remove
                else:
                    raise ValueError(f"Unknown score_mode: {self.score_mode!r}")
                
            if best_to_remove is not None:
                active.pop(best_to_remove)
                full_score = best_candidate_score
                improved = True

        # Rebuild ar_lag and pred_lags from remaining active pairs
        new_pred_lags = np.zeros_like(pred_lags)
        new_ar_lag = 0
        for j, lag in active:
            if j == target_idx:
                new_ar_lag = lag
                new_pred_lags[j] = lag  # keep AR lag on diagonal
            else:
                new_pred_lags[j] = lag

        return new_ar_lag, new_pred_lags

    def fit(self, X: np.ndarray):
        """
        Fit lag selection for all variables as targets.

        For each variable i (row), the method:
        - selects its best AR lag,
        - selects best lag of every variable j as predictor of i (including the possibility of 0 = not used).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input multivariate time series.

        Returns
        -------
        ar_lags : ndarray of shape (n_features,)
            Best AR lag per variable (0..max_lag).
        pred_lag_matrix : ndarray of shape (n_features, n_features)
            Matrix of selected lags:
            pred_lag_matrix[i, j] is the lag of variable j used to predict i
            (0 means j is not used as a lagged predictor for i).
        """
        X_proc = self._preprocess(X)
        T, D = X_proc.shape

        # Determine which targets to process
        if self.target_indices is None:
            targets = list(range(D))
        else:
            targets = list(self.target_indices)

        # Parallel initial pairwise search
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_pair_for_target)(X_proc, target_idx=i)
            for i in targets
        )

        ar_lags = np.zeros(D, dtype=int)
        pred_lag_matrix = np.zeros((D, D), dtype=int)
        
        for idx, i in enumerate(targets):
            best_ar_lag, best_pred_lags = results[idx]
            ar_lags[i] = best_ar_lag
            if best_ar_lag > 0:
                best_pred_lags[i] = best_ar_lag
            pred_lag_matrix[i, :] = best_pred_lags

        # Optional pruning phase
        if self.prune_lags:
            new_ar = np.zeros_like(ar_lags)
            new_pred = np.zeros_like(pred_lag_matrix)
            for i in targets:
                ar_i, pred_i = self._prune_lags_for_target(
                    X_proc,
                    target_idx=i,
                    ar_lag=ar_lags[i],
                    pred_lags=pred_lag_matrix[i, :]
                )
                new_ar[i] = ar_i
                new_pred[i, :] = pred_i
            ar_lags, pred_lag_matrix = new_ar, new_pred

        return ar_lags, pred_lag_matrix

    @staticmethod
    def max_lag_per_pred(pred_lag_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the maximum lag per predictor variable (column-wise).

        Parameters
        ----------
        pred_lag_matrix : ndarray of shape (n_features, n_features)
            Matrix of selected lags: element [i, j] is lag j->i.

        Returns
        -------
        max_lags : ndarray of shape (n_features,)
            For each predictor j, maximum lag used across all targets i.
        """
        return pred_lag_matrix.max(axis=0)

    @staticmethod
    def build_weight_mask(pred_lag_matrix: np.ndarray,
                          use_lag_zero: bool = False):
        """
        Build a binary mask for linear regression weights A in a model:

            Y = A @ X_lagged + B

        Columns of X_lagged are constructed by concatenating, for each
        predictor j:

            - optionally lag 0 (current value) if use_lag_zero=True,
            - all lags 1..max_lag_j, where max_lag_j is the maximum selected
              lag for predictor j across all targets.

        IMPORTANT
        ---------
        pred_lag_matrix[i, j] is interpreted as a *maximum lag* for target i
        and predictor j:
            - pred_lag_matrix[i, j] = L > 0  means: allow lags 1..L of
              predictor j when predicting target i.
            - pred_lag_matrix[i, j] = 0      means: predictor j is not used
              for target i (no lagged values).

        Parameters
        ----------
        pred_lag_matrix : ndarray of shape (n_targets, n_features)
            pred_lag_matrix[i, j] = maximum lag of feature j used to predict
            target i (0 means feature j is not used for target i).
        use_lag_zero : bool, default False
            If True, allocate one extra column per predictor for lag=0
            (current value) and set mask[:, col_lag0_j] = 1 for all targets.

        Returns
        -------
        mask : ndarray of shape (n_targets, total_lag_features)
            Binary mask for A:
            - mask[i, k] = 1 if weight A[i, k] is allowed to be optimized,
              0 if it should be fixed to zero.
            - Columns are ordered as blocks per predictor j:
              if use_lag_zero is True:
                  [lag0_j] + [lag1_j, ..., lag_max_j]
              otherwise:
                  [lag1_j, ..., lag_max_j]
        max_lags_per_pred : ndarray of shape (n_features,)
            max_lags_per_pred[j] = max selected lag for predictor j
            across all targets (can be 0).
        col_offsets : ndarray of shape (n_features,)
            col_offsets[j] = starting column index of predictor j block
            in the mask and in the corresponding X_lagged design matrix.
            Note: if use_lag_zero is True, col_offsets[j] points to lag0_j.
        """
        pred_lag_matrix = np.asarray(pred_lag_matrix, dtype=int)
        n_targets, n_features = pred_lag_matrix.shape

        # Maximum lag per predictor (column-wise)
        max_lags_per_pred = pred_lag_matrix.max(axis=0)  # shape (n_features,)

        # Column offsets and total number of columns
        col_offsets = np.zeros(n_features, dtype=int)
        total_cols = 0
        for j in range(n_features):
            col_offsets[j] = total_cols
            # number of columns for predictor j:
            # lag0 (if used) + lags 1..max_lag_j
            n_cols_j = max_lags_per_pred[j]
            if use_lag_zero:
                n_cols_j += 1
            total_cols += n_cols_j

        # Initialize mask
        mask = np.zeros((n_targets, total_cols), dtype=int)

        for j in range(n_features):
            start_col = col_offsets[j]
            max_L_j = max_lags_per_pred[j]

            # Optional lag0 column: always allowed for all targets
            if use_lag_zero:
                col_lag0 = start_col
                mask[:, col_lag0] = 1  # all targets can use x_t^{(j)}

            if max_L_j <= 0:
                continue  # no positive lags for this predictor in any target

            # Columns for lags 1..max_L_j
            # If use_lag_zero=True, lag1 starts at start_col+1,
            # otherwise lag1 starts at start_col.
            base = start_col + (1 if use_lag_zero else 0)

            for i in range(n_targets):
                L = pred_lag_matrix[i, j]
                if L <= 0:
                    continue  # predictor j not used for target i (for lags >0)
                L_eff = min(L, max_L_j)
                # Set mask 1 for lags 1..L_eff (local indices 0..L_eff-1 in this block)
                mask[i, base:base + L_eff] = 1

        return mask, max_lags_per_pred, col_offsets
