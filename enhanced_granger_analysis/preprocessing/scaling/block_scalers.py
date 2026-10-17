from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from .registry import resolve_scaler
from ...core.exceptions import DataShapeError, ScalerNotFittedError

def _as_2d_float64(data: NDArray[np.float64]) -> NDArray[np.float64]:
	"""Return a float64 2D array copy suitable for numerical scaling."""
	arr = np.asarray(data, dtype=np.float64)
	if arr.ndim != 2:
		raise DataShapeError(
			f"Expected 2D array of shape (n_samples, n_features), got ndim={arr.ndim}"
		)
	return arr.copy()

class BlockScaler:
    """Apply one fitted scalar transformation to each predictor block."""

    def __init__(
        self,
        blocks: dict[str, slice],
        scaler: Any = "standard",
    ):
        self.blocks = dict(blocks)
        self.scaler = scaler
        self.scalers_: dict[str, Any] = {}
        self.n_features_: Optional[int] = None

    def _validate_blocks(self, n_features: int) -> None:
        coverage = np.zeros(n_features, dtype=int)

        for name, block in self.blocks.items():
            if not isinstance(block, slice):
                raise ValueError(f"Block {name!r} must be a slice.")

            if (
                block.start is None
                or block.stop is None
                or block.step not in (None, 1)
                or not isinstance(block.start, (int, np.integer))
                or not isinstance(block.stop, (int, np.integer))
                or not 0 <= block.start < block.stop <= n_features
            ):
                raise ValueError(
                    f"Invalid block {name!r}: {block!r}."
                )

            coverage[block] += 1

        if not np.all(coverage == 1):
            raise ValueError(
                "Blocks must cover every feature exactly once."
            )

    def fit(self, data: NDArray[np.float64]) -> "BlockScaler":
        import copy

        x = _as_2d_float64(data)
        if x.shape[0] == 0:
            raise ValueError("Cannot fit BlockScaler on empty data.")

        self._validate_blocks(x.shape[1])
        fitted = {}

        for name, block in self.blocks.items():
            block_scaler = copy.deepcopy(resolve_scaler(self.scaler))
            values = x[:, block].reshape(-1, 1)

            # The accepted scaler contract does not require fit().
            block_scaler.fit_transform(values)
            fitted[name] = block_scaler

        self.scalers_ = fitted
        self.n_features_ = x.shape[1]
        return self

    def _apply(
        self,
        data: NDArray[np.float64],
        method_name: str,
    ) -> NDArray[np.float64]:
        if self.n_features_ is None:
            raise ScalerNotFittedError(
                "BlockScaler is not fitted."
            )

        x = _as_2d_float64(data)
        if x.shape[1] != self.n_features_:
            raise DataShapeError(
                f"Expected {self.n_features_} features, "
                f"got {x.shape[1]}."
            )

        result = x.copy()

        for name, block in self.blocks.items():
            shape = x[:, block].shape
            values = x[:, block].reshape(-1, 1)
            transform = getattr(self.scalers_[name], method_name)

            transformed = np.asarray(
                transform(values),
                dtype=np.float64,
            )
            if transformed.shape != values.shape:
                raise DataShapeError(
                    f"Scaler for block {name!r} changed the data shape."
                )

            result[:, block] = transformed.reshape(shape)

        return result

    def transform(
        self,
        data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._apply(data, "transform")

    def inverse_transform(
        self,
        data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._apply(data, "inverse_transform")

    def fit_transform(
        self,
        data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.fit(data).transform(data)