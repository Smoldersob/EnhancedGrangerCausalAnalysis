from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from ...core.protocols import Scaler, Regularizer, Constraint

class BaseGrangerModel(ABC):
    """
    Abstract base class for Granger causality models with pluggable components.
    
    This class defines the interface for Granger causality models supporting:
    - Multiple backends (statsmodels, sklearn, custom)
    - External scalers (ETS in/out-of-sample)
    - External regularizers (Lasso, Ridge, etc.)
    - External constraints (positivity, sparsity, etc.)
    """
    
    def __init__(
        self,
        backend: str = 'default',
        scaler: Optional[Scaler] = None,
        regularizer: Optional[Regularizer] = None,
        constraint: Optional[Constraint] = None
    ) -> None:
        """
        Initialize the Granger causality model with pluggable components.
        
        Args:
            backend: Backend identifier ('sklearn', 'keras', 'custom', etc.)
            scaler: Scaler instance implementing Scaler protocol
            regularizer: Regularizer instance implementing Regularizer protocol  
            constraint: Constraint instance implementing Constraint protocol
        """
        self.backend: str = backend
        self.scaler: Optional[Scaler] = scaler
        self.regularizer: Optional[Regularizer] = regularizer
        self.constraint: Optional[Constraint] = constraint
        self._weights: List[NDArray[np.float64]] = []
        self._fitted: bool = False

    @abstractmethod
    def initialize(
        self, 
        data: NDArray[np.float64], 
        lags: int, 
        **kwargs
    ) -> None:
        """
        Initialize model with data and hyperparameters.
        
        Uses injected scaler if provided during preprocessing.
        
        Args:
            data: Time series data array of shape (n_samples, n_features)
            lags: Number of lags for Granger causality test
            **kwargs: Additional initialization parameters
        """
        pass

    @abstractmethod
    def fit(self) -> Dict[str, Any]:
        """
        Fit the model using injected regularizer and constraints.
        
        Returns:
            Dictionary containing fit results:
                - 'test_statistic': Granger causality test statistic
                - 'p_value': Test p-value
                - 'weights': Fitted model weights
                - 'forecasts': Model forecasts
        """
        pass

    @abstractmethod
    def set_weights(self, weights: Union[NDArray[np.float64], List[NDArray[np.float64]]]) -> None:
        """
        Set model weights with optional constraint enforcement.
        
        Args:
            weights: Model weights array or list of weight matrices
        """
        pass

    def get_weights(self) -> List[NDArray[np.float64]]:
        """
        Get fitted model weights.
        
        Returns:
            List of weight matrices for each lag
        """
        return self._weights

    def get_backend(self) -> str:
        """
        Get current backend identifier.
        
        Returns:
            Backend name string
        """
        return self.backend

    def set_scaler(self, scaler: Scaler) -> None:
        """Dynamically set scaler component."""
        self.scaler = scaler

    def set_regularizer(self, regularizer: Regularizer) -> None:
        """Dynamically set regularizer component."""
        self.regularizer = regularizer

    def set_constraint(self, constraint: Constraint) -> None:
        """Dynamically set constraint component."""
        self.constraint = constraint

    @abstractmethod
    def hyperoptimize(
        self,
        reg_param_grid: Dict[str, List[Any]],
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Hyperparameter optimization for regularization parameters.
        
        Args:
            reg_param_grid: Dictionary of regularization parameter grids
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimization results:
                - 'best_params': Best regularization parameters
                - 'best_score': Best optimization score
                - 'trial_results': All trial results
        """
        pass