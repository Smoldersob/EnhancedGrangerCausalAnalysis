import numpy as np
import pandas as pd

from ..utilits import static_adfuller_order
from ..lag_engine import auto_select_lag,make_static
from statsmodels.tsa.stattools import grangercausalitytests

def grangers_causation_matrix(data, causes=None, effects=None, test='ssr_chi2test', lag=None, lag_max=20):    
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    Parameters
    ----------
    data : pd.DataFrame 
        Table containing the time series variables
    causes : list 
        containing names of the time series which might be causes (predictors)
    effects : list 
        List containing names of the time series which might be affected by causes (response)
    test : string 
        Name of statistic test type (ssr_ftest,ssr_chi2test,lrtest,params_ftest)
    max_lag : int, optional (default=20)
        Maximum number of time lags to consider in the model.

    Return
    --------------
    pd.DataFrame
        Table of 0-1 indicating if the cause singal can or can not cause changes in th other signal.
    """
    for name in data.columns:
        order=static_adfuller_order(data[name])
        data[name]=make_static(data=data[name],order=order)
    data=data.dropna()

    #Lag control
    if lag is None:
        lag_order=auto_select_lag(data,max_lag=lag_max)   
    else:
        lag_order=lag

    #Analised data control
    if causes is None or causes is []:
        causes=data.columns.to_list()
    if effects is None or effects is []:
        effects=data.columns.to_list()
        
    df = pd.DataFrame(np.zeros((len(effects), len(causes))), columns=causes, index=effects)
    

    for c in causes:
        for r in effects:
            test_result = grangercausalitytests(data[[r, c]], maxlag=[lag_order], verbose=False)
            p_values = test_result[lag_order][0][test][1]
            min_p_value = np.min(p_values)

            #Sign
            max_coef=test_result[lag_order][1][1].params[lag_order:-1].max()
            min_coef=test_result[lag_order][1][1].params[lag_order:-1].min()
            
            df.loc[r, c] = (min_p_value<0.01)*1
            if df.loc[r, c]==1:
                df.loc[r, c]*=np.sign(max_coef+min_coef)
    df.columns = [str(var) + '_x' for var in causes]
    df.index = [str(var) + '_y' for var in effects]
    return df