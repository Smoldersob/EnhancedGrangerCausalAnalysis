from abc import abstractmethod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Scaler(Protocol):
    @abstractmethod
    def fit(self, data: NDArray[np.float64]) -> "Scaler": ...

    @abstractmethod
    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def inverse_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...
