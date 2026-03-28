"""
Defines protocols for scalers, regularizers, constraints, and the Granger model interface.
These protocols enable flexible integration of various components into Granger causality models,
allowing for modular design and easy experimentation with different preprocessing, regularization, and constraint strategies.
"""

from abc import abstractmethod
from typing import Any, Dict, Protocol
import numpy as np
from numpy.typing import NDArray

class Scaler(Protocol):
    """Interface for data scaling methods (ETS in/out-of-sample)."""
    
    @abstractmethod
    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Fit the scaler and transform the input data.
        
        Args:
            data: Input time series data array of shape (n_samples, n_features)
            
        Returns:
            Transformed data array with the same shape as input
        """
        ...
    
    @abstractmethod
    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform new input data using the fitted scaler.
        
        Args:
            data: Input time series data array of shape (n_samples, n_features)
            
        Returns:
            Transformed data array with the same shape as input
        """
        ...

    @abstractmethod
    def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Inverse transform the data back to original scale.
        
        Args:
            data: Scaled data array of shape (n_samples, n_features)
            
        Returns:
            Original scale data array with the same shape as input
        """
        ... 


class Regularizer(Protocol):
    """Interface for regularization methods."""
    
    @abstractmethod
    def apply(self, model_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply regularization to model parameters.
        
        Args:
            model_params: Model parameter array
            
        Returns:
            Regularized parameter array
        """
        ...
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get current regularization parameters.
        
        Returns:
            Dictionary containing regularization parameters
        """
        ...


class Constraint(Protocol):
    """Interface for parameter constraints."""
    
    @abstractmethod
    def enforce(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Enforce constraints on model parameters.
        
        Args:
            params: Parameter array to constrain
            
        Returns:
            Constrained parameter array
        """
        ...
    
    @abstractmethod
    def is_satisfied(self, params: NDArray[np.float64]) -> bool:
        """
        Check if constraints are satisfied for given parameters.
        
        Args:
            params: Parameter array to check
            
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        ...

class GrangerModelProtocol(Protocol):
    """Protocol defining the interface for Granger causality models."""
    
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Fit the Granger model to the provided data.
        
        Args:
            X: Input feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        pass

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generate predictions from the fitted model.
        
        Args:
            X: Input feature array of shape (n_samples, n_features)
            
        Returns:
             Predictions array of shape (n_samples,)
        """
        pass