import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def static_kpss_order(series: pd.Series, maxlag: int = 3) -> pd.Series:
    """
    Iteratively differences a time series until it becomes stationary according to the KPSS test.

    Parameters
    --------------
    series : pd.Series
        Input time series data to test for stationarity.

    Return
    --------------
    pd.Series
        Differenced series that is KPSS-stationary (p-value ≥ 0.05).

    Notes:
    ------
    - Uses the KPSS test with trend/constant regression (regression='ct').
    - Repeatedly applies first-order differencing (series - series.shift(1)) until stationarity is achieved.
    - Drops NaN values before testing.
    """
    series=series.copy().dropna().values
    if series.var()==0:
        return 0
    i_best=0
    p_best=np.inf
    for i in range(maxlag+1):
        adf_res=kpss(series, regression='ct')
        if adf_res[1]<=0.05:
            i_best=i
            break
        else:
            if p_best>adf_res[1]:
                i_best=i
                p_best=adf_res[1]
        series=np.roll(series,0,axis=0)[1:]-np.roll(series,1,axis=0)[1:]
        
    return i_best

def static_adfuller_order(series: pd.Series, maxlag: int = 3) -> pd.Series:
    """
    Iteratively differences a time series until it becomes stationary according to the ADF test.

    Parameters
    --------------
    series : pd.Series
        Input time series data to test for stationarity.
    maxlag : int, optional
        Maximum lag order for the ADF test (default=20).

    Return
    --------------
    pd.Series
        Differenced series that is ADF-stationary (p-value ≤ 0.05).

    Notes:
    ------
    - Uses the Augmented Dickey-Fuller (ADF) test.
    - Applies first-order differencing (series - series.shift(1)) until stationarity is achieved.
    - Drops NaN values before testing.
    """

    series=series.copy().dropna().values
    if series.var()==0:
        return 0
    
    i_best=0
    p_best=np.inf
    for i in range(maxlag+1):
        adf_res=adfuller(series,maxlag=maxlag)
        if adf_res[1]<=0.05:
            i_best=i
            break
        else:
            if p_best>adf_res[1]:
                i_best=i
                p_best=adf_res[1]
        series=np.roll(series,0,axis=0)[1:]-np.roll(series,1,axis=0)[1:]
        
    return i_best