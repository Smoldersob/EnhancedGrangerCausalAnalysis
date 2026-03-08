import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR


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