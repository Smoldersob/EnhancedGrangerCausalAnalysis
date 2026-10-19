import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, Dict, List, Mapping, Optional, Union

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
        regularizer: Any = None,
        constraint: Any = None,
        callbacks: Optional[List[Any]] = None,
        needs_reinit: bool = True,
    ):
        """
        Initialize the Granger causality model with pluggable components.
        
        Args:
            backend: Backend identifier ('sklearn', 'keras', 'custom', etc.)
            regularizer: A ready backend regularizer instance or None. 
            constraint: Constraint instance implementing Constraint protocol
            callbacks: List of callbacks implementing callback protocol
        """
        self.backend: str = backend
        self.regularizer: Any = regularizer
        self.constraint: Any = constraint
        self.callbacks = list(callbacks) if callbacks is not None else []
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
    def get_weights(self) -> List[NDArray[np.float64]]:
        """Return independent copies of model coefficient arrays.

        The concrete model defines the number, order and shapes of arrays.
        Input-control masks and optimizer state are excluded.
        """
        raise NotImplementedError


    @abstractmethod
    def set_weights(
        self,
        weights: List[NDArray[np.float64]],
    ) -> None:
        """Assign model coefficients without changing input-control state.

        Validate all arrays before assignment.
        Do not retain references to the supplied arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def omit_variables(self, variable_indices: List[int]) -> None:
        """Replace the omitted input set without changing model coefficients.

        Each call starts from basic state.
        An empty puts it in to basic state.
        Basic state of models models currently is all inputs enabled, but is not guaranteed for future models.
        Current model coefficients are preserved, but the omitted inputs are ignored in predictions.
        """
        raise NotImplementedError

    @staticmethod
    def _validated_weight_arrays(
        weights: List[NDArray[np.float64]],
        expected_shapes: List[tuple[int, ...]],
    ) -> List[NDArray[np.float64]]:
        if not isinstance(weights, list):
            raise ValueError("weights must be a list of arrays.")

        if len(weights) != len(expected_shapes):
            raise ValueError(
                f"Expected {len(expected_shapes)} weight arrays, "
                f"got {len(weights)}."
            )

        validated = []

        for index, (value, shape) in enumerate(
            zip(weights, expected_shapes)
        ):
            array = np.array(value, dtype=np.float64, copy=True)

            if array.shape != shape:
                raise ValueError(
                    f"Weight {index} has shape {array.shape}, "
                    f"expected {shape}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Weight {index} contains non-finite values."
                )

            validated.append(array)

        return validated

    def get_backend(self) -> str:
        """
        Get current backend identifier.
        
        Returns:
            Backend name string
        """
        return self.backend

    @abstractmethod
    def reset_callbacks(
        self,
        run_name: str,
        callback_updates: Optional[
            Mapping[Union[int, str, type], Mapping[str, Any]]
        ] = None,
    ) -> None:
        """Prepare callbacks for an independent training run.

        Preserve callback configuration unless explicitly updated.
        Discard callback runtime state from previous training.

        Selectors in callback_updates may identify callbacks by:
        - index,
        - class name,
        - callback type.

        Do not modify model coefficients, input masks or optimizer state.
        Call between fit() calls, not during training.
        """
        raise NotImplementedError