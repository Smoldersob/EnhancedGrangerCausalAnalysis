import pandas as pd
import numpy as np
import copy
import random
import datetime
from typing import List,Dict

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .complex_granger import ComplexGrangerAnalisysModel
from ..models.PytorchSparseLinearModel import SparseLinearModel,RelationExists
from ..callbacks.callbacks import EarlyStopping,ProcentageChange
from ..granger_analysis_results import RSS
from ..regularizers.regularizers_pytorch import CyclicL1Regularizer

class PTNeuralSparseConstaraintedMVGC(ComplexGrangerAnalisysModel):
    regularizer=CyclicL1Regularizer()
    
    def __init__(
            self,
            max_lag:int = 20,
            learning_rate:float|None = None,
            relative_referece_learning_rate:float = 0.05,
            batch_size:int = 32,
            epochs:int = 1000,
            referece_epochs:int = 1000,
            sparse:float = 0.0,
            auto_sparse_iterations:int = 20,
            sparse_fit_epochs:int = 100,
            writer = None,
            writer_outdir:str = "logs/fit/",
            optimizer=optim.Adam,
            scheduler=None,
            **kwargs
        ):
        super().__init__(auto_sparse_iterations = auto_sparse_iterations,
                         max_lag = max_lag,**kwargs)
        self.optimizer=optimizer
        self.scheduler=scheduler
        
        self.learning_rate = learning_rate
        self.relative_referece_learning_rate = relative_referece_learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.referece_epochs=referece_epochs
        
        self.sparse=sparse
        self.sparse_fit_epochs = sparse_fit_epochs

        self.writer = writer
        self.writer_outdir = writer_outdir

        self.verbose=False
        
    def _select_optimal_l1_param(self, X, y, constraint=None, callbacks=[], seed=None):
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
            model = SparseLinearModel(input_dim=X_train.shape[1],
                                      output_dim=y_train.shape[1],
                                      constraint=constraint,
                                      lasso = alpha,
                                      batch_size=self.batch_size,
                                      epochs=self.sparse_fit_epochs,
                                      x_norm_mean=self.mean1,
                                      x_norm_std=self.std1,
                                      y_norm_mean=self.mean2,
                                      y_norm_std=self.std2,
                                      seed=seed)
        
            
            opt=self.optimizer(model.parameters(), lr=self.learning_rate)
            # Fit model
            model.train_model(X_train=X_train,
                              y_train=y_train,
                              optimizer=opt,
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
            base_lag:int = None,
            custom_lag: Dict[str,List[int]] = {},
            callbacks=[EarlyStopping()],
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
        base_lag : int, optional
            Number of lagged time steps to include. If None, lag order is selected automatically.
        custom_lag : dict, optional
            Dictionary of lag ranges for column given by key. If value consists of of list of 2 elements first they are
            treated as lowest and largest lag used on column. If there is list with one value ist is treated  as largest lag.  
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
        >>> model = NeuralSparseConstaraintedMVGC(max_lag=5, sparse=0.1, learning_rate=0.01, epochs=100)
        >>> model.fit(data=df, causes=['X1', 'X2'], effects=['Y'], lag=3, seed=42)
        >>> model.results.result()
        """

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
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
        Xs, y, column_indexes =  super().prepare_lag(data_list=data_list_static,effects=effects,lag=base_lag,custom_lag=custom_lag)
        if self.verbose: print(f"{self.lag_order}")
        Xs, y, forced_relation, possible_relation = super().prepare_experts_knowladge(Xs=Xs,y=y,columns=columns_names,
                                                                                      effects=effects,relation=relation,
                                                                                      column_indexes=column_indexes,
                                                                                      seed=seed,unused_data=unused_data)
        
        
        x_l = Xs.shape[1]  
        constraint=RelationExists(possible_relation,forced_relation) 

        if hasattr(self.regularizer,'set_lag_orders'):
            self.regularizer.set_lag_orders(column_indexes)
            
        #Auto learning rate
        if self.learning_rate is None:
            self.learning_rate = 0.5/x_l
        
        if self.batch_size<=0:
            self.batch_size = Xs.shape[0]

        #Writer
        if self.writer:
            writer = SummaryWriter(self.writer_outdir+"base_model_pytorch_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            writer=self.writer

        #Normalization and denormalization parameters
        self.mean1 = Xs.mean(axis=0)
        self.mean2 = -y.mean(axis=0)/y.std(axis=0)
        self.std1 = Xs.std(axis=0)
        self.std2 = 1/y.std(axis=0)

        #Sparsity control
        if self.sparse < 0:
            self.sparse = self._select_optimal_l1_param(X=Xs,y=y,
                                                        constraint=constraint,
                                                        callbacks=callbacks,
                                                        seed=seed)
            if self.verbose:print(f'L1 value: {self.sparse}')
            
        alfa = float(self.sparse)    
        #Base model
        SparseLinearModel.regularizer=self.regularizer

        modelall = SparseLinearModel(input_dim=x_l,
                                     output_dim=nrows,
                                     constraint=constraint,
                                     lasso = alfa,
                                     batch_size=self.batch_size,
                                     epochs=self.epochs,
                                     x_norm_mean=self.mean1,
                                     x_norm_std=self.std1,
                                     y_norm_mean=self.mean2,
                                     y_norm_std=self.std2,
                                     seed=seed)

        opt=self.optimizer(modelall.parameters(), lr=self.learning_rate)
        if self.scheduler is not None:
            scheduler = copy.deepcopy(self.scheduler)
            scheduler.optimizer=opt
        else:
            scheduler = None
        
        modelall.train_model(X_train=Xs,y_train=y,
                             optimizer=opt,
                             callbacks=[copy.deepcopy(cb) for cb in callbacks],
                             tensoboard_writer=writer,
                             scheduler=scheduler,
                             verbose=self.verbose)
        
        #Data lacking models
        lr = self.learning_rate*self.relative_referece_learning_rate
        possible_relation_mod=possible_relation.copy()
        
        for nr,name in zip(columns_id,causes):
            if self.verbose: print('\n',name)
            possible_relation_mod[:,column_indexes[nr]:column_indexes[nr+1]]=0

            if self.writer:
                writer = SummaryWriter(self.writer_outdir+"reference_model_"+name+"_pytorch_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

            constraint2=RelationExists(relation_table=possible_relation_mod,relation_list=forced_relation)
            modelmissone = SparseLinearModel(input_dim=x_l,
                                         output_dim=nrows,
                                         constraint=constraint2,
                                         lasso = alfa,
                                         batch_size=self.batch_size,
                                         epochs=self.referece_epochs,
                                         x_norm_mean=self.mean1,
                                         x_norm_std=self.std1,
                                         y_norm_mean=self.mean2,
                                         y_norm_std=self.std2,
                                         seed=seed)
            
            if self.verbose: print("Copying weights")
            modelmissone.set_weights(modelall.linear.weight.data.clone(), modelall.linear.bias.data.clone())
            
            opt2=self.optimizer(modelmissone.parameters(), lr=lr)
            if self.scheduler is not None:
                scheduler = copy.deepcopy(self.scheduler)
                scheduler.optimizer=opt2
            else:
                scheduler = None

            modelmissone.train_model(X_train=Xs,y_train=y,
                             optimizer=opt2,
                             callbacks=[copy.deepcopy(cb) for cb in callbacks],
                             tensoboard_writer=writer,
                             verbose=self.verbose)
        
            self.results.update_column(name, column_id=nr,
                                       base_model = modelall, ref_model = modelmissone,
                                       x = Xs,y = y,
                                       column_indexes = column_indexes,
                                       model_type=2)
            
            possible_relation_mod[:,column_indexes[nr]:column_indexes[nr+1]]=possible_relation[:,column_indexes[nr]:column_indexes[nr+1]]
        