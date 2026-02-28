import pandas as pd
import numpy as np
import copy
import random
import datetime
from typing import List, Dict

import tensorflow as tf
from tensorflow import random as tfrandom 
gpus = tf.config.list_physical_devices('GPU') 
if gpus: 
    try: 
        tf.config.experimental.set_memory_growth(gpus[0], True)
        device_name = '/GPU:0' 
    except RuntimeError as e:
        print(f"GPU setup failed: {e}. Falling back to CPU.") 
        device_name = '/CPU:0' 
else: 
    device_name = '/CPU:0'

from keras import Sequential
from keras.layers import Dense,Input,Normalization
from keras.callbacks import EarlyStopping,TensorBoard, ReduceLROnPlateau
from keras.constraints import Constraint
from keras.regularizers import L1
from keras.optimizers import Adam

from .complex_granger import ComplexGrangerAnalisysModel
from ..granger_analysis_results import RSS        
from ..regularizers.regularizers_keras import KerasCyclicL1Regularizer
from ..models.MaskedDenseLayer import MaskedDense

class TFNeuralSparseConstaraintedMVGC(ComplexGrangerAnalisysModel):
    """
    Neural network-based Sparse Constrained Multivariate Granger Causality (MVGC) model.

    This class implements a neural network approach to estimate Granger causality relationships
    in multivariate time series data, incorporating sparsity constraints and lagged dependencies.
    It supports configurable training parameters including learning rates, batch size, epochs,
    and optimization algorithm. The model can automatically tune sparsity parameters and optionally
    log training progress.

    Parameters
    ----------
    lag_max : int, optional (default=20)
        Maximum number of time lags to consider in the model.
    learning_rate : float or None, optional (default=None)
        Base learning rate for model training. If None, defaults are set by the optimizer.
    relative_referece_learning_rate : float, optional (default=0.1)
        Learning rate multiplier for reference models relative to the base model.
    batch_size : int, optional (default=32)
        Number of samples per batch during training.
    epochs : int, optional (default=1000)
        Number of training iterations.
    sparse : float, optional (default=0.0)
        Sparsity regularization strength controlling L1 penalty.
    auto_sparse_iterations : int, optional (default=20)
        Number of iterations for automatic sparsity parameter tuning.
    sparse_fit_epochs : int, optional (default=1000)
        Number of training iterations (epochs) for automatic regularization parameter fitting.
    writer : bool or object, optional (default=False)
        Enables logging of training progress if True or a logging object.
    writer_outdir : str, optional (default="logs/fit/")
        Directory path for saving training logs.
    optimizer : tf.keras.optimizers.Optimizer, optional (default=Adam())
        Optimizer instance used for training the neural network.

    Attributes
    ----------
    lag_max : int
        Maximum lag order.
    learning_rate : float or None
        Learning rate for training.
    relative_referece_learning_rate : float
        Learning rate multiplier for reference models.
    batch_size : int
        Training batch size.
    epochs : int
        Number of training epochs.
    referece_epochs : int
        Number of training epochs for reference models.
    sparse : float
        Sparsity regularization parameter.
    auto_sparse_iterations : int
        Iterations for automatic sparsity tuning.
    sparse_fit_epochs : int
        Number of training iterations (epochs) for automatic regularization parameter fitting.
    writer : bool or object
        Logger for training progress.
    writer_outdir : str
        Output directory for logs.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer used for training.
    results : GA_results
        Container for storing fitted model results.

    Notes
    -----
    - The model leverages neural networks for flexible modeling of nonlinear causal dependencies.
    - GPU acceleration and cross-platform compatibility may be leveraged for training efficiency[1][2].
    - The class is designed to integrate with callbacks and logging utilities for monitoring training.

    Examples
    --------
    >>> model = NeuralSparseConstaraintedMVGC(lag_max=15, learning_rate=0.001, epochs=500)
        >>> model.fit(data=df, causes=['X1', 'X2'], effects=['Y'], seed=42)
    """
    regularizer=KerasCyclicL1Regularizer()

    def __init__(
            self,
            max_lag:int = 20,
            learning_rate:float = None,
            relative_referece_learning_rate:float = 0.05,
            batch_size:int = None,
            epochs:int = 1000,
            referece_epochs:int = 1000,
            sparse:float = 0.0,
            auto_sparse_iterations:int = 20,
            sparse_fit_epochs:int = 100,
            writer = False,
            writer_outdir:str = "logs/fit/",
            optimizer = Adam(),
            **kwargs
        ):
        super().__init__(auto_sparse_iterations = auto_sparse_iterations,
                         max_lag = max_lag, **kwargs)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.relative_referece_learning_rate = relative_referece_learning_rate
        self.referece_epochs=referece_epochs

        self.writer = writer
        self.writer_outdir = writer_outdir
        
        self.sparse=sparse
        self.sparse_fit_epochs = sparse_fit_epochs

        self.optimizer = optimizer
        self._base_optimizer_config = None
        self._ref_optimizer_config = None

        self.verbose=False


    def _select_optimal_l1_param(self, X, y, possible_relation, forced_relation, callbacks=[], seed=None):
        """
        Select optimal L1 regularization parameter for a tensorflow.keras model.
        (Should work like CV-Lasso).
        Parameters:
            X : array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
           
            y : array-like, shape (n_samples,)
                Target vector relative to X.

            constraint : RelationExists
                Object of RelationExists class that is constraint for Granger Analisys relations. 

            callback : list-like, [tf.keras.Callback], optional
                List of callbacks for ANN.
            
            seed : intiger, optional
                Seed for reproducibility.
            
        Returns:
            best_alpha: float
                The optimal regularization parameter value found
        """

        X_train, X_val, y_train, y_val, alphas, best_score, best_alpha = super().prepare_data_for_l1_fitting(X, y, seed)

        masked_dense = MaskedDense(
            y.shape[1],
            kernel_regularizer=self.regularizer,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            forced_relations=forced_relation
        )

        model = Sequential([
            Input(shape=(X.shape[1],)),
            Normalization(mean=self.mean1, variance=self.var1, dtype='float32', name='NormX'),
            masked_dense,
            Normalization(mean=self.mean2, variance=self.var2, dtype='float32', name='NormY')])
    
        cv_optimizer = self._create_or_reset_optimizer(self.learning_rate if self.learning_rate else 0.001, is_base=True)
        masked_dense.update_mask(possible_relation.T.astype('float32'))
        model.compile(loss="mean_squared_error", optimizer=cv_optimizer)
        
        initial_weights = model.get_weights()
        
        for alpha in alphas:
            self.regularizer.l1 = alpha
            model.set_weights(initial_weights)
            self._reset_optimizer_state(model=model)
        
            cv_callbacks = self._create_callbacks(callbacks, log_suffix=None)
            
            # Fit model
            model.fit(x = X_train, y = y_train,
                    epochs = self.sparse_fit_epochs,
                    batch_size = self.batch_size,
                    verbose = self.verbose,
                    callbacks = cv_callbacks)
            
            # Evaluate on validation set
            score = np.mean(RSS(model, X_val, y_val))
            
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        return best_alpha

    def _create_callbacks(self, callbacks_template, log_suffix=None):
        """Create fresh callback instances from template list."""
        new_callbacks = []
        for cb in callbacks_template:
            if isinstance(cb, EarlyStopping):
                new_callbacks.append(EarlyStopping(
                    monitor=cb.monitor,
                    patience=cb.patience,
                    start_from_epoch=cb.start_from_epoch,
                    min_delta=cb.min_delta
                ))
            else:
                new_callbacks.append(copy.deepcopy(cb))
        
        if self.writer and log_suffix:
            log_dir = self.writer_outdir + log_suffix + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            new_callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
        
        return new_callbacks

    def _reset_optimizer_state(self, model):
        """Reset optimizer internal state without changing learning rate."""

        for var in model.optimizer.variables:
            if not var.path.__contains__('learning_rate'):
                var.assign(tf.zeros_like(var))
            
    def _create_or_reset_optimizer(self, learning_rate, is_base=True):
        """Create optimizer instance or reset existing one with new learning rate."""
        config_key = '_base_optimizer_config' if is_base else '_ref_optimizer_config'
        
        if getattr(self, config_key) is None:
            opt_config = self.optimizer.get_config()
            setattr(self, config_key, opt_config.copy())
        
        config = getattr(self, config_key).copy()
        config['learning_rate'] = learning_rate
        
        return self.optimizer.__class__.from_config(config)

    def fit(
            self,
            data: pd.DataFrame|List[pd.DataFrame],
            causes:list = None,
            effects:list = None,
            relation:dict = dict(),
            base_lag:int = None,
            custom_lag: Dict[str,List[int]] = {},
            callbacks = [EarlyStopping(monitor = 'loss', patience = 15,start_from_epoch=1,min_delta=1e-8),
                         ReduceLROnPlateau(monitor = 'loss', patience = 7,min_delta=1e-8)],
            seed = None,
            unused_data = 0
        ):
        """
        Train the neural sparse constrained MVGC model on multivariate time series data.

        This method preprocesses the input data by ensuring stationarity using the Augmented Dickey-Fuller test,
        selects or uses a specified lag order to create lagged features, and constructs constraints based on known
        causal relations. It then trains a base neural network model with L1 sparsity regularization and multiple
        reference models excluding one cause at a time to assess causal influence robustness. Training progress can
        be logged via TensorBoard and controlled with callbacks.

        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series data with columns as variables.
        causes : list, optional
            List of variable names considered as causes. Defaults to all columns if None or empty.
        effects : list, optional
            List of variable names considered as effects. Defaults to all columns if None or empty.
        relation : dict, optional
            Dictionary specifying known causal relations as keys (cause, effect) and values indicating relation type
            (e.g., 0 to forbid causality). Used to build constraints on model weights.
        base_lag : int, optional
            Number of lagged time steps to include. If None, lag order is selected automatically.
        custom_lag : dict, optional
            Dictionary of lag ranges for column given by key. If value consists of of list of 2 elements first they are
            treated as lowest and largest lag used on column. If there is list with one value ist is treated  as largest lag.  
        callbacks : list, optional
            List of Keras callback instances for training control (e.g., early stopping).
        seed : int, optional
            Random seed for reproducibility.
        unused_data : [0,1], optional
            Value used as test_size for train_test_split, allowing to used only some part of data instead of all data.

        Returns
        -------
        None
            Fitted models and results are stored internally in `self.results`.

        Details
        -------
        - Applies stationarity transformation to each variable using `make_static_adfuller`.
        - Automatically selects lag order if not provided, using `auto_select_lag`.
        - Constructs lagged datasets for input features and targets.
        - Builds constraints on model weights based on known relations to enforce or forbid causal links.
        - If learning rate is unspecified, sets it inversely proportional to number of features.
        - Uses L1 regularization to enforce sparsity; can auto-tune sparsity parameter if negative.
        - Normalizes inputs and outputs for stable training.
        - Trains a base model with all causes included.
        - Trains reference models excluding one cause at a time to evaluate its causal effect.
        - Supports logging training metrics with TensorBoard if enabled.
        - Uses callbacks such as early stopping to prevent overfitting.
        - Resets optimizer state between base and reference model training.
    
        Notes
        -----
        - The method relies on TensorFlow and Keras for neural network training[1][2].
        - The `RelationExists` constraint enforces structural causal assumptions during training.
        - The `GA_results` object collects and organizes model fitting outputs for analysis.

        Examples
        --------
        >>> model = NeuralSparseConstaraintedMVGC(max_lag=10, epochs=500, sparse=0.1)
        >>> model.fit(data=df, causes=['X1', 'X2'], effects=['Y'], seed=42)
        """

        if seed is not None:
            tfrandom.set_seed(seed)
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
        Xs, y, column_indexes =  super().prepare_lag(data_list=data_list_static,effects=effects,lag=base_lag,custom_lag=custom_lag)
        if self.verbose: print(f"{self.lag_order}")
        Xs, y, forced_relation, possible_relation = super().prepare_experts_knowladge(Xs=Xs,y=y,columns=columns_names,
                                                                                      effects=effects,relation=relation,
                                                                                      column_indexes=column_indexes,
                                                                                      seed=seed,unused_data=unused_data)
        
        x_l = Xs.shape[1]  
        
        if hasattr(self.regularizer,'set_lag_orders'):
            self.regularizer.set_lag_orders(column_indexes)
        

        #Auto learning rate
        if self.learning_rate is None:
            self.learning_rate = 0.5/x_l
        
        if self.batch_size is not None and self.batch_size<=0:
            self.batch_size = Xs.shape[0]

        #Normalization and denormalization parameters
        self.mean1 = Xs.mean(axis=0)
        self.mean2 = -y.mean(axis=0)/y.std(axis=0)
        self.var1 = Xs.var(axis=0)
        self.var2 = 1/y.var(axis=0)

        #Sparsity control
        if self.sparse < 0:
            self.sparse = self._select_optimal_l1_param(X=Xs,y=y,
                                                        possible_relation=possible_relation,
                                                        forced_relation=forced_relation,
                                                        callbacks=callbacks,seed=seed)
            if self.verbose:print(f'L1 value: {self.sparse}')
            
        if hasattr(self.regularizer,'l1'): self.regularizer.l1 = float(self.sparse)
        if self.verbose: print(f"Learning rate: {self.learning_rate}")
        
        # Fresh optimizer and callbacks
        base_optimizer = self._create_or_reset_optimizer(self.learning_rate, is_base=True)
        callbacks_base = self._create_callbacks(callbacks, log_suffix="base_model_keras")
    
        #Base model
        masked_dense_base = MaskedDense(
            nrows,
            kernel_regularizer=self.regularizer,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            forced_relations=forced_relation
        )
    
        modelall = Sequential([
            Input(shape = (Xs.shape[1],)),
            Normalization(mean = self.mean1, variance = self.var1, dtype='float32', name = 'NormX'),
            masked_dense_base,
            Normalization(mean = self.mean2, variance = self.var2, dtype='float32', name = 'NormY')])
        
        masked_dense_base.update_mask(possible_relation.T.astype('float32'))
        
        modelall.compile(loss="mean_squared_error", optimizer = base_optimizer)

        modelall.fit(x = Xs, y = y,
                    epochs = self.epochs,
                    batch_size = self.batch_size,
                    verbose = self.verbose,
                    callbacks = callbacks_base)
    
        base_weights = modelall.get_weights()

        # Reference models optimizer and constaints
        ref_lr = self.learning_rate * self.relative_referece_learning_rate
    
        #Reference model
        masked_dense_ref = MaskedDense(
            nrows,
            kernel_regularizer=self.regularizer,
            use_bias=True,
            forced_relations=forced_relation
        )
        
        modelmissone = Sequential([
                Input(shape = (Xs.shape[1],)),
                Normalization(mean = self.mean1, variance = self.var1, dtype='float32', name = 'NormX2'),
                masked_dense_ref,
                Normalization(mean = self.mean2, variance = self.var2, dtype='float32', name = 'NormY2')])

        ref_optimizer = self._create_or_reset_optimizer(ref_lr, is_base=False)
        modelmissone.compile(loss="mean_squared_error", optimizer=ref_optimizer)

        possible_relation_working = possible_relation.copy().astype('float32')
        
        for nr,name in zip(columns_id,causes):
            if self.verbose: print(name)

            possible_relation_working[:, column_indexes[nr]:column_indexes[nr+1]] = 0
            if name!=causes[0]: self._reset_optimizer_state(modelmissone)

            #Reset
            if self.verbose: print("Copying weights")
            ref_weights=base_weights[:-1]
            ref_weights.append(possible_relation_working.T)
            modelmissone.set_weights(ref_weights)

            callbacks_ref = self._create_callbacks(callbacks, log_suffix=f"reference_model_{name}_keras")
            
            modelmissone.fit(x = Xs, y = y,
                                epochs = self.referece_epochs,
                                batch_size = self.batch_size,
                                verbose = self.verbose,
                                callbacks = callbacks_ref)

            self.results.update_column(name, column_id=nr,
                                       base_model = modelall, ref_model = modelmissone,
                                       x = Xs,y = y,
                                       column_indexes = column_indexes,
                                       model_type=1)
            
            possible_relation_working[:, column_indexes[nr]:column_indexes[nr+1]] = possible_relation[:, column_indexes[nr]:column_indexes[nr+1]]
        
        