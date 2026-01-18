import pandas as pd
import numpy as np
from typing import List
from joblib import Parallel, delayed

from ..utilits import static_adfuller_order
from ..lag_engine import (
    auto_select_lag,
    create_lagged_data,
    make_static
)
from ..granger_analisys_results import GrangerAnalisysResults

from sklearn.model_selection import train_test_split

class ComplexGrangerAnalisysModel():
    
    n_jobs=1
    with_zero_lag=False

    def __init__(self,
                 auto_sparse_iterations:int,
                 max_lag:int,
                 non_static:List[str]=[],
                 **kwargs
                ):
        self.auto_sparse_iterations = auto_sparse_iterations
        self.max_lag = max_lag
        self.verbose = False
        self.non_static=non_static
        self.results = GrangerAnalisysResults([],[])

    def _select_optimal_l1_param(self, X, y, constraint=None, callbacks=[], seed=None):
        pass

    def fit(
            self,
            data: pd.DataFrame|List[pd.DataFrame],
            causes: list = None,
            effects: list = None,
            relation: dict = dict(),
            lag: int = None,
            callbacks=[],
            seed=None,
            unused_data=0
        ):
        """
        Fit the model to the provided time series data, estimating causal relationships with lagged effects.

        This method preprocesses the input data by applying stationarity transformations, selects or uses a specified lag order,
        constructs lagged datasets, and fits a base model along with reference models that account for missing causal inputs.
        It supports incorporating prior knowledge about causal relations and enforces sparsity constraints during model fitting.
        Optionally, it logs training progress using TensorBoard writers and supports callback functions during optimization.

        Parameters
        ----------
        data : pd.DataFrame
        Time series data with columns representing variables. Each column should be a univariate time series.
        causes : list, optional
            List of variable names to be considered as potential causes. If None or empty, all columns in `data` are used.
        effects : list, optional
            List of variable names to be considered as effects (dependent variables). If None or empty, all columns in `data` are used.
        relation : dict, optional
            Dictionary specifying known causal relations between variables as keys (tuples of cause and effect) and values indicating
            the type of relation (e.g., 0 to enforce no causal effect). Used to constrain model coefficients accordingly.
        lag : int, optional
            The number of lagged time steps to include in the model. If None, the lag order is selected automatically.
        callbacks : list, optional
            List of callback instances to be called during model training, e.g., for monitoring or early stopping.
        seed : int, optional
            Random seed for reproducibility of results.
        unused_data : [0,1], optional
            Value used as test_size for train_test_split, allowing to used only some part of data instead of all data.

        Return
        -------
        None
            The method stores the fitted models and results internally in `self.results`.

        """
        pass

    def prepare_data_for_l1_fitting(self, X, y, seed=None):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
        # Generate a list of candidate alphas on a log scale (e.g., from 1e-4 to 1)
        alphas = np.logspace(-6, 0, self.auto_sparse_iterations)
        best_score = np.inf
        alphas = alphas.tolist()
        best_alpha = 0
        alphas = [0] + alphas
        return X_train, X_val, y_train, y_val, alphas, best_score, best_alpha
     
    def prepare_static(self,
                data_list: List[pd.DataFrame],
                causes: list = None,
                effects: list = None,
            ):
        
        #Analised data control
        columns_names=data_list[0].columns.to_list()
        columns_id=[data_list[0].columns.to_list().index(cause) for cause in causes]
    
        nrows=len(effects)
        self.results = GrangerAnalisysResults(effects = effects,causes = causes)
        
        static_orders=[0 for i in range(len(columns_names))]
            
        for data in data_list:
            #Presupposition check
            results=[]
            for var in data.columns:
                if var in self.non_static:
                    results.append(0)
                else:
                    results.append(static_adfuller_order(data[var], self.max_lag))
            static_orders=np.max([static_orders,results],axis=0)
        self.static_orders=static_orders

        def force_static(data,static_orders):
            data_c=pd.DataFrame()
            data_c=data.copy()
            for order,name in zip(static_orders,data.columns):
                data_c[name] = make_static(data[name], order = order)
            data_c=data.dropna()    
            return data_c
        
        tasks = []
        for data in data_list:
            tasks.append((data, static_orders))
        
        data_list_static = Parallel(n_jobs=self.n_jobs)(
            delayed(force_static)(*task) for task in tasks
        )

        return nrows, columns_id, data_list_static
    
    def prepare_lag(self,
                data_list: List[pd.DataFrame],
                effects: list = None,
                lag: int = None,
                ):
    
        if lag is None:
            tasks = []
            for data in data_list:
                tasks.append((data, self.max_lag))
            
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(auto_select_lag)(*task) for task in tasks
            )
            lag_order = np.max(results)
        else:
            lag_order = lag
        self.lag_order=lag_order

        if self.with_zero_lag: lag_order=lag_order+1

        tasks = []
        for data in data_list:
            tasks.append((data, lag_order, self.with_zero_lag))
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(create_lagged_data)(*task) for task in tasks
        )    
        
        Xs=np.concat(results,axis=0)
        
        def drop_unusable_ys(data, effects, order):
            return data[effects].iloc[order:].values

        tasks2 = []
        for data in data_list:
            tasks2.append((data, effects, lag_order))
        
        results2 = Parallel(n_jobs=self.n_jobs)(
            delayed(drop_unusable_ys)(*task) for task in tasks2
        )    
        
        y=np.concat(results2,axis=0)
        return Xs, y, lag_order
    
    def prepare_experts_knowladge(
            self,
            Xs:pd.DataFrame|np.ndarray,
            y:pd.DataFrame|np.ndarray,
            columns: list = None,
            effects: list = None,
            relation: dict = dict(),
            lag_order: int = 0,
            seed=None,
            unused_data=0
        ):

        #Using only part of data
        if unused_data:
            Xs, X_v, y, y_v = train_test_split(Xs, y, test_size=unused_data, random_state=seed)
            X_v=None
            y_v=None

        #Phicical-information coding
        nrows=len(effects)
        x_l = Xs.shape[1]
        
        possible_relation=np.ones((nrows,x_l))
        forced_relation=[]
        for c1,c2 in relation.keys():
            if c2 in effects:
                i = effects.index(c1)
                if c2 in columns:
                    j = columns.index(c2)*lag_order
                    if relation[(c1,c2)] == 0:
                        possible_relation[i,j:lag_order+j] = 0
                    else:
                        forced_relation.append([i,[k for k in range(j,lag_order+j)],relation[(c1,c2)]]) 
        
        if  self.with_zero_lag:
            for c in effects:
                i = effects.index(c)
                j = columns.index(c)*lag_order
                possible_relation[i,j] = 0                    

        return Xs, y, forced_relation, possible_relation
