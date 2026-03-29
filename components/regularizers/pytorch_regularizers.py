from __future__ import annotations

import importlib
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


if find_spec("torch") is not None:
	torch = importlib.import_module("torch")
	nn = importlib.import_module("torch.nn")
else:  # pragma: no cover - runtime dependency check
	torch = None
	nn = None


def _ensure_torch() -> None:
	if torch is None:
		raise RuntimeError("PyTorch is required to use pytorch_regularizers.")


class PyTorchL1Regularizer(nn.Module if nn is not None else object):
	"""Standard L1 regularizer for PyTorch models."""

	def __init__(self, l1: float = 0.0) -> None:
		_ensure_torch()
		super().__init__()
		if l1 < 0:
			raise ValueError("l1 must be >= 0")
		self.l1 = float(l1)

	def forward(self, x: Any) -> Any:
		target_dtype = x.dtype if torch.is_tensor(x) else torch.float32
		x_cast = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=target_dtype)
		if self.l1 == 0.0:
			return torch.tensor(0.0, dtype=target_dtype, device=x_cast.device)
		return torch.tensor(self.l1, dtype=target_dtype, device=x_cast.device) * torch.sum(torch.abs(x_cast))

	def apply(self, model_params: NDArray[np.float64]) -> NDArray[np.float64]:
		params = np.asarray(model_params, dtype=np.float64)
		return self.l1 * np.sign(params)

	def get_params(self) -> Dict[str, Any]:
		return {"l1": self.l1}


class PyTorchLagDependentL1Regularizer(nn.Module if nn is not None else object):
	"""Lag-dependent L1 regularizer for PyTorch models.

	For tensors with rank >= 2, lag weights are applied on the last axis
	(which corresponds to input features for nn.Linear weights).
	"""

	_DEFAULT_INITIAL_VALUES: List[float] = [
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    ]

	def __init__(
		self,
		l1: float = 0.0,
		lag_weights: Optional[Sequence[float]] = None,
		max_lags_per_pred: Optional[Sequence[int]] = None,
		col_offsets: Optional[Sequence[int]] = None,
	) -> None:
		_ensure_torch()
		super().__init__()

		if l1 < 0:
			raise ValueError("l1 must be >= 0")
		self.l1 = float(l1)

		if lag_weights is None:
			lag_weights = list(self._DEFAULT_INITIAL_VALUES)
		if len(lag_weights) == 0:
			raise ValueError("lag_weights cannot be empty")
		self.lag_weights = [float(v) for v in lag_weights]

		if (max_lags_per_pred is None) != (col_offsets is None):
			raise ValueError("Provide both max_lags_per_pred and col_offsets, or neither.")

		self.max_lags_per_pred = None
		self.col_offsets = None
		if max_lags_per_pred is not None and col_offsets is not None:
			self.set_lag_layout(max_lags_per_pred=max_lags_per_pred, col_offsets=col_offsets)

	def set_lag_layout(
		self,
		max_lags_per_pred: Sequence[int],
		col_offsets: Sequence[int],
	) -> None:
		"""Assign lag layout produced by LagSelector/LagEngine."""
		if len(max_lags_per_pred) == 0:
			raise ValueError("max_lags_per_pred cannot be empty")
		if len(max_lags_per_pred) != len(col_offsets):
			raise ValueError("max_lags_per_pred and col_offsets must have equal length")
		if any(int(v) < 0 for v in max_lags_per_pred):
			raise ValueError("All max_lags_per_pred values must be >= 0")

		offsets = [int(v) for v in col_offsets]
		if offsets[0] != 0:
			raise ValueError("col_offsets must start at 0")
		if any(offsets[i] > offsets[i + 1] for i in range(len(offsets) - 1)):
			raise ValueError("col_offsets must be non-decreasing")

		self.max_lags_per_pred = [int(v) for v in max_lags_per_pred]
		self.col_offsets = offsets

	def _feature_weights_np(self, n_features: int) -> NDArray[np.float64]:
		"""Build per-feature weights using LagSelectionResult block layout."""
		if self.max_lags_per_pred is not None and self.col_offsets is not None:
			if len(self.max_lags_per_pred) != len(self.col_offsets):
				raise ValueError("max_lags_per_pred and col_offsets lengths must match")

			offsets = list(self.col_offsets)
			ends = offsets[1:] + [int(n_features)]
			per_feature = np.empty(n_features, dtype=np.float64)

			for j, (start, end) in enumerate(zip(offsets, ends)):
				if start < 0 or end < start or end > n_features:
					raise ValueError("Invalid col_offsets for current number of features")

				max_lag = int(self.max_lags_per_pred[j])
				block_len = int(end - start)
				if block_len == 0:
					continue

				if block_len > (max_lag + 1):
					raise ValueError(
						f"Predictor block {j} has length {block_len}, expected <= {max_lag + 1} "
						f"from max_lags_per_pred"
					)

				lag_start_idx = max_lag - block_len + 1
				if lag_start_idx < 0:
					raise ValueError(
						f"Cannot infer lag start for block {j}: max_lag={max_lag}, "
						f"block_len={block_len}"
					)

				for rel_idx in range(block_len):
					lag_idx = lag_start_idx + rel_idx
					if lag_idx >= len(self.lag_weights):
						raise ValueError(
							f"lag_weights length ({len(self.lag_weights)}) is too short for "
							f"lag index {lag_idx}"
						)
					per_feature[start + rel_idx] = self.lag_weights[lag_idx]

			return per_feature

		if len(self.lag_weights) == n_features:
			return np.asarray(self.lag_weights, dtype=np.float64)

		if len(self.lag_weights) == 1:
			return np.full(n_features, self.lag_weights[0], dtype=np.float64)

		raise ValueError(
			"Unable to map lag_weights to features. Set max_lags_per_pred and col_offsets, "
			"or provide lag_weights per feature."
		)

	def forward(self, x: Any) -> Any:
		x_tensor = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)
		if self.l1 == 0.0:
			return torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)

		if x_tensor.dim() == 0:
			return torch.tensor(self.l1, dtype=x_tensor.dtype, device=x_tensor.device) * torch.abs(x_tensor)

		n_features = int(x_tensor.shape[-1])
		feature_weights = self._feature_weights_np(n_features)
		feature_weights_t = torch.as_tensor(feature_weights, dtype=x_tensor.dtype, device=x_tensor.device)

		broadcast_shape = [1] * (x_tensor.dim() - 1) + [n_features]
		feature_weights_t = feature_weights_t.reshape(broadcast_shape)

		weighted_abs = torch.abs(x_tensor) * feature_weights_t
		return torch.tensor(self.l1, dtype=x_tensor.dtype, device=x_tensor.device) * torch.sum(weighted_abs)

	def apply(self, model_params: NDArray[np.float64]) -> NDArray[np.float64]:
		params = np.asarray(model_params, dtype=np.float64)
		if params.ndim == 0:
			return np.asarray(0.0, dtype=np.float64)

		n_features = params.shape[-1]
		feature_weights = self._feature_weights_np(n_features)
		reshape = (1,) * (params.ndim - 1) + (n_features,)
		feature_weights = feature_weights.reshape(reshape)
		return self.l1 * np.sign(params) * feature_weights

	def get_params(self) -> Dict[str, Any]:
		return {
			"l1": self.l1,
			"lag_weights": list(self.lag_weights),
			"max_lags_per_pred": (
				list(self.max_lags_per_pred)
				if self.max_lags_per_pred is not None
				else None
			),
			"col_offsets": list(self.col_offsets) if self.col_offsets is not None else None,
		}
