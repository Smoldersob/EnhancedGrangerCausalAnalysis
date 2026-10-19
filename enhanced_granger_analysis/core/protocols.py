"""
Defines protocols for scalers and initializers.
These protocols enable flexible integration of various components into Granger causality models,
allowing for modular design and easy experimentation with different preprocessing, regularization, and constraint strategies.
"""

from abc import abstractmethod
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

from .initialization import InitializationContext, WeightArray

class Scaler(Protocol):
    """Interface for data scaling methods (ETS in/out-of-sample)."""

    @abstractmethod
    def fit(self, data: NDArray[np.float64]) -> "Scaler":
        """
        Fit parameter of scaler.

        Args:
            data: Input time series data array of shape (n_samples, n_features)

        Returns:
            Scaler object with fitted parameters
        """

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


class Initializer(Protocol):
    """Interface for linear model weight initialization."""

    def __call__(
        self,
        *,
        context: InitializationContext,
    ) -> list[WeightArray]:
        """Return weight arrays for each target."""
        ...
        