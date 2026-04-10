from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
from ...core.protocols import Regularizer, Constraint

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
        regularizer: Optional[Regularizer] = None,
        constraint: Optional[Constraint] = None,
        callbacks: List[Any] = [],
        needs_reinit: bool = True,
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
        self.regularizer: Optional[Regularizer] = regularizer
        self.constraint: Optional[Constraint] = constraint
        self.callbacks: List[Any] = callbacks
        # If True, caller should run initialize(...) before each training cycle.
        self.needs_reinit: bool = bool(needs_reinit)
        self._fitted: bool = False

    @abstractmethod
    def initialize(
        self, 
        data: NDArray[np.float64], 
        lags: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize model with data and hyperparameters.
        
        Uses injected scaler if provided during preprocessing.
        
        Args:
            data: Time series data array of shape (n_samples, n_features)
            lags: Optional lag hint from preprocessing (not required by backend models)
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
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generate predictions using fitted model parameters.

        Args:
            X: Input feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted values array of shape (n_samples, n_outputs)
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
    
    @abstractmethod
    def get_weights(self) -> List[NDArray[np.float64]]:
        """
        Get fitted model weights.
        
        Returns:
            List of weight matrices for each lag
        """
        pass

    @abstractmethod
    def omit_variables(self, variable_indices: List[int]) -> None:
        """
        Omit specified variables from the model (e.g., for testing).
        
        Args:
            variable_indices: List of variable indices to omit
        """
        pass

    def get_backend(self) -> str:
        """
        Get current backend identifier.
        
        Returns:
            Backend name string
        """
        return self.backend

    def _set_regularizer(self, regularizer: Regularizer) -> None:
        """Dynamically set regularizer component."""
        self.regularizer = regularizer

    def _set_constraint(self, constraint: Constraint) -> None:
        """Dynamically set constraint component."""
        self.constraint = constraint

    def _add_callback(self, callback: Any) -> None:
        """Add a callback to the model's training process."""
        self.callbacks.append(callback)