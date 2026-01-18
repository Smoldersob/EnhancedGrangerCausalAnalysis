import numpy as np
import pandas as pd
import copy
from typing import List
import datetime
import random

from tensorflow import summary as SummaryWriter

from .complex_granger import ComplexGrangerAnalisysModel
from ..models.MultiTaskConstrainedLinearRegression import MultiTaskConstrainedLinearRegression as MTCLR
from ..callbacks.callbacks import ProcentageChange
from ..granger_analisys_results import RSS

class SparseConstaraintedMVGC(ComplexGrangerAnalisysModel):
    """
    Sparse Constrained Multivariate Granger Causality (MVGC) model class.

    This class implements a multivariate Granger causality model with sparsity constraints and lagged dependencies.
    It supports training with gradient-based optimization, configurable learning rates, batch sizes, and epoch counts.
    The model can automatically select sparsity parameters through iterative procedures and optionally log training progress.

    Parameters
    ----------
    max_lag : int, optional (default=20)
        Maximum number of time lags to consider in the model.
    learning_rate : float, optional (default=1)
        Base learning rate for model optimization.
    relative_referece_learning_rate : float, optional (default=1.0)
        Relative learning rate multiplier for reference models compared to the base model.
    batch_size : int or None, optional (default=None)
        Size of batches for training optimization. If None, full-batch training is used.
    epochs : int, optional (default=1000)
        Number of training iterations (epochs).
    sparse : float, optional (default=0.0)
        Sparsity regularization parameter controlling L1 penalty strength.
    auto_sparse_iterations : int, optional (default=20)
        Number of iterations for automatic sparsity parameter selection.
    sparse_fit_epochs : int, optional (default=1000)
        Number of training iterations (epochs) for automatic regularization parameter fitting. 
    writer : bool or object, optional (default=False)
        If True or a writer object, enables logging of training progress (e.g., TensorBoard).
    writer_outdir : str, optional (default="logs/fit/")
        Directory path where training logs will be saved if writer is enabled.

    Attributes
    ----------
    max_lag : int
        Maximum lag order used in the model.
    learning_rate : float
        Learning rate for optimization.
    relative_referece_learning_rate : float
        Multiplier for learning rate in reference models.
    batch_size : int or None
        Batch size for training.
    epochs : int
        Number of training epochs.
    sparse : float
        Sparsity regularization parameter.
    auto_sparse_iterations : int
        Iterations for auto sparsity tuning.
    writer : bool or object
        Logger for training progress.
    writer_outdir : str
        Output directory for logs.
    results : GA_results
        Container for storing model fitting results.

    Examples
    --------
    >>> model = SparseConstaraintedMVGC(max_lag=10, learning_rate=0.5, epochs=500, sparse=0.1)
    >>> model.fit(data=df, causes=['X1', 'X2'], effects=['Y'], lag=3, seed=42)
    >>> model.results.result()
    """

    def __init__(
            self,
            max_lag:int = 20,
            learning_rate:float = 1,
            relative_referece_learning_rate:float = 1.,
            batch_size:int = None,
            epochs:int = 1000,
            sparse:float = 0.0,
            auto_sparse_iterations:int = 20,
            sparse_fit_epochs:int = 30,
            writer = False,
            writer_outdir:str = "logs/fit/",
            cycle_lasso=False,
            **kwargs
        ):
        super().__init__(auto_sparse_iterations = auto_sparse_iterations,
                         max_lag = max_lag,**kwargs)
        
        self.learning_rate = learning_rate
        self.relative_referece_learning_rate = relative_referece_learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.sparse=sparse
        self.sparse_fit_epochs = sparse_fit_epochs
        self.cycle_lasso=cycle_lasso

        self.writer = writer
        self.writer_outdir = writer_outdir

        self.verbose = False
    
    def _select_optimal_l1_param(self, X, y, min_coefs=None, max_coefs=None, forced_relation=None, callbacks=[], seed=None):
        """
        Select optimal L1 regularization parameter for a MultiTaskConstrainedLinearRegression2D
        (should work for Lasso scikit-learn-like models).
        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
           
            y : array-like, shape (n_samples,)
                Target vector relative to X.

            min_coefs : array-like, shape (n_features,), optional
                Lower constraint for coefficients. Defaults to negative infinity for each coefficient.

            max_coefs : array-like, shape (n_features,), optional
                Upper constraint for coefficients. Defaults to positive infinity for each coefficient.

            forced_relation : dictlist-like, [int,list/range-like,float], optional
                List containing task number (int), coefficients positions (list/range-like) and value (flaot). The value is
                the smallest acceptable sum of absolute values of chosen coefficents of chosen task.
    
            callback : list-like, [MTCLR_Callback], optional
                List of callbacks. Used call backs have to fit/derive from MTCLR_Callback class in order to work properly.
            
            seed : intiger, optional
                Seed for reproducibility.
            
        Returns:
            best_alpha: float
                The optimal regularization parameter value found
        """
        X_train, X_val, y_train, y_val, alphas, best_score, best_alpha = super().prepare_data_for_l1_fitting(X, y, seed)

        for alpha in alphas:
            # Instantiate the model with L1 penalty and current alpha
            model = MTCLR(lasso = alpha,
                      learning_rate = self.learning_rate,
                      max_iter = self.sparse_fit_epochs)
            
            # Fit model
            model.fit(X=X_train,y=y_train,
                     min_coef=min_coefs, max_coef=max_coefs,
                     abs_sum_min=forced_relation,
                     batch=self.batch_size,
                     tensoboard_writer=False,
                     callbacks=[copy.deepcopy(cb) for cb in callbacks])
            
            # Evaluate on validation set
            score = np.mean(RSS(model, X_val, y_val))
            
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        return best_alpha

    def fit(
            self,
            data: pd.DataFrame|List[pd.DataFrame],
            causes: list = None,
            effects: list = None,
            relation: dict = dict(),
            lag: int = None,
            callbacks = [ProcentageChange()],
            seed = None,
            unused_data = 0
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

        Notes
        -----
        - The method applies the Augmented Dickey-Fuller test transformation to ensure stationarity of each time series.
        - If sparsity parameter (`self.sparse`) is negative, an optimal L1 regularization parameter is selected automatically.
        - The base model is trained with all causes included, while reference models are trained excluding one cause at a time to assess its impact.
        - TensorBoard writers are created if `self.writer` is enabled, to log training metrics.
        - The method updates `self.results` with the fitted models and related metadata for further analysis.

        Raises
        ------
        ValueError
            If input data is invalid or if lag order selection fails.

        Examples
        --------
        >>> model = SparseConstaraintedMVGC(max_lag=5, sparse=0.1, learning_rate=0.01, epochs=100)
        >>> model.fit(data=df, causes=['X1', 'X2'], effects=['Y'], lag=3, seed=42)
        >>> model.results.result()
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if type(data)==type(pd.DataFrame()):
            data=[data]
        elif type(data)==type([]):
            pass
        else:
            raise TypeError("Data should be dataframe or list of dataframes")

        columns_names=data[0].columns.to_list()
        if causes is None or causes is []:
            causes = columns_names
        if effects is None or effects is []:
            effects = columns_names

        nrows, columns_id, data_list_static = super().prepare_static(data_list=data,causes=causes,effects=effects)        
        if self.verbose: print("Set lag:")
        Xs, y, lag_order =  super().prepare_lag(data_list=data_list_static,effects=effects,lag=lag)
        if self.verbose: print(f"{self.lag_order}")
        Xs, y, forced_relation, possible_relation = super().prepare_experts_knowladge(Xs=Xs,y=y,columns=columns_names,
                                                                                      effects=effects,relation=relation,
                                                                                      lag_order=lag_order,
                                                                                      seed=seed,unused_data=unused_data)
        
        x_l = Xs.shape[1]
        min_coefs = -np.inf**possible_relation+1
        max_coefs = -min_coefs
        
        #Writer
        if self.writer:
            writer = SummaryWriter.create_file_writer(self.writer_outdir+"base_model_scikit_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            writer=self.writer

        #Sparsity control
        if self.sparse < 0:
            self.sparse = self._select_optimal_l1_param(X=Xs,y=y,
                                                        min_coefs=min_coefs, max_coefs=max_coefs,
                                                        forced_relation=forced_relation,
                                                        callbacks=callbacks, seed=seed)
        alfa = float(self.sparse)    
        cycle=0
        if self.cycle_lasso: cycle=lag_order

        #Base model
        modelall = MTCLR(lasso = alfa,
                      learning_rate = self.learning_rate,
                      max_iter = self.epochs,
                      cycle_period=cycle)
        if self.verbose: print("Training base model")
        modelall.fit(X=Xs,y=y,
                     min_coef=min_coefs, max_coef=max_coefs,
                     abs_sum_min=forced_relation,
                     batch=self.batch_size, tensoboard_writer=writer,
                     callbacks=[copy.deepcopy(cb) for cb in callbacks])
        
        #Data lacking models
        lr = self.learning_rate*self.relative_referece_learning_rate
        modelmissone = MTCLR(lasso = alfa,
                          learning_rate = lr,
                          max_iter = self.epochs,
                          cycle_period=cycle)
        min_coefs_add = min_coefs.copy()
        max_coefs_add = max_coefs.copy()
        for nr,name in zip(columns_id,causes):
            if self.verbose: print(name)
            min_coefs_add[:,[nr*lag_order+_ for _ in range(lag_order)]]=0
            max_coefs_add[:,[nr*lag_order+_ for _ in range(lag_order)]]=0

            if self.writer:
                writer = SummaryWriter.create_file_writer(self.writer_outdir+"reference_model_"+name+"_scikit_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            if self.verbose:
                print(f"Training reference model without {name}")
            modelmissone.fit(X = Xs, y = y,
                             min_coef = min_coefs_add, max_coef = max_coefs_add,
                             abs_sum_min = forced_relation,
                             batch = self.batch_size, tensoboard_writer = writer,
                             callbacks=[copy.deepcopy(cb) for cb in callbacks],
                             initial_beta = modelall.coef_)
        
            self.results.update_column(name, column_id=nr,
                                       base_model = modelall, ref_model = modelmissone,
                                       x = Xs,y = y,
                                       lag_order = lag_order)
            
            min_coefs_add[:,[nr*lag_order+_ for _ in range(lag_order)]] = min_coefs[:,[nr*lag_order+_ for _ in range(lag_order)]]
            max_coefs_add[:,[nr*lag_order+_ for _ in range(lag_order)]] = max_coefs[:,[nr*lag_order+_ for _ in range(lag_order)]]
