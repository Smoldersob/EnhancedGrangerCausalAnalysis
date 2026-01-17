import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR


def auto_select_lag(data: pd.DataFrame, max_lag: int = 20, maximum: bool = True) -> int:
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
    p = np.arange(1,max_lag)
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
    
def create_lagged_data(data: pd.DataFrame, lag: int, with_zero=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates lagged features and aligned target variables for time-series modeling.

    Parameters
    --------------
    data : pd.DataFrame
        Input time series data (columns = features).
    lag : int
        Number of lag periods to create.

    Return
    --------------
    X : np.array
        Lagged feature matrix with shape (n_samples, n_features * lag). 

    Notes:
    ------
    - Drops initial rows with falsely values due to lagging.
    - Columns in X are sorted hierarchically by signal name and lag value (ascending).
    """
    
    x=data.values.copy()
    if with_zero:
        X=np.zeros((x.shape[0],x.shape[1]*lag))
        for k in range(x.shape[1]):
            X[:,k*lag:k*lag+lag]=np.concat([np.roll(x[:,[k]],i,axis=0) for i in range(lag)],axis=1)
        X=X[lag:,:]
    else:
        x=data.values.copy()
        X=np.zeros((x.shape[0],x.shape[1]*lag))
        for k in range(x.shape[1]):
            X[:,k*lag:k*lag+lag]=np.concat([np.roll(x[:,[k]],i+1,axis=0) for i in range(lag)],axis=1)
        X=X[lag:,:]
    return X

def make_static(data,order:int):
    if order:
        for i in range(0,order):
            data=np.roll(data,0,axis=0)-np.roll(data,1,axis=0)
        data[:order+1]=np.nan
        #data[-order:]=np.nan
        return data
    else:
        return data