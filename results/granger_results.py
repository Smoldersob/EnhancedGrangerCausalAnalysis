from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .causality_matrix import CausalityMatrices
from .statistics import ensure_2d, error_and_p_values


@dataclass
class ModelSnapshot:
	"""Stored outputs from one fitted model instance."""

	predictions: NDArray[np.float64]
	weights: NDArray[np.float64]


class GrangerAnalysisResults:
	"""
	Container and calculator for multitask Granger analysis outputs.

	This class stores base/reference model predictions and weights and updates
	all causality matrices for each tested cause.
	"""

	def __init__(self, effects: Iterable[str], causes: Iterable[str]) -> None:
		self.effects: List[str] = list(effects)
		self.causes: List[str] = list(causes)
		self.matrices = CausalityMatrices.create(self.effects, self.causes)

		self.base_snapshot: Optional[ModelSnapshot] = None
		self.reference_snapshots: Dict[str, ModelSnapshot] = {}

	def _extract_output_feature_weights(
		self,
		model: object,
		n_outputs: int,
		n_features: int,
	) -> NDArray[np.float64]:
		"""Convert model.get_weights() output into shape (n_outputs, n_features)."""
		if not hasattr(model, "get_weights"):
			raise TypeError("Model must define get_weights()")

		weights_list = model.get_weights()
		if not isinstance(weights_list, list) or len(weights_list) == 0:
			raise ValueError("model.get_weights() must return a non-empty list")

		kernel = np.asarray(weights_list[0], dtype=np.float64)
		if kernel.ndim != 2:
			raise ValueError(f"Expected 2D kernel matrix, got shape {kernel.shape}")

		# Preferred convention in current models: (n_features, n_outputs)
		if kernel.shape == (n_features, n_outputs):
			return kernel.T
		# Fallback if backend returns (n_outputs, n_features)
		if kernel.shape == (n_outputs, n_features):
			return kernel

		raise ValueError(
			"Cannot infer weight orientation from kernel shape "
			f"{kernel.shape}; expected {(n_features, n_outputs)} or {(n_outputs, n_features)}"
		)

	@staticmethod
	def _sign_from_block(weight_block: NDArray[np.float64]) -> NDArray[np.float64]:
		"""
		Compute sign per output using coefficient with maximum absolute value.

		For each output row, this matches requirement #sym:sign.
		"""
		if weight_block.ndim != 2:
			raise ValueError("weight_block must be 2D")
		if weight_block.shape[1] == 0:
			return np.zeros(weight_block.shape[0], dtype=np.float64)

		argmax = np.argmax(np.abs(weight_block), axis=1)
		selected = weight_block[np.arange(weight_block.shape[0]), argmax]
		return np.sign(selected).astype(np.float64)

	def update_cause(
		self,
		cause: str,
		cause_index: int,
		base_model: object,
		reference_model: object,
		X: NDArray[np.float64],
		y: NDArray[np.float64],
		col_offsets: NDArray[np.int_],
	) -> None:
		"""Update all matrices and snapshots for one tested cause variable."""
		if cause not in self.causes:
			raise ValueError(f"Unknown cause: {cause}")
		if cause_index < 0 or cause_index + 1 >= len(col_offsets):
			raise ValueError("cause_index out of range for provided col_offsets")

		y_true = ensure_2d(np.asarray(y, dtype=np.float64))
		X_arr = ensure_2d(np.asarray(X, dtype=np.float64))
		n_outputs = y_true.shape[1]
		n_features = X_arr.shape[1]

		base_pred = ensure_2d(np.asarray(base_model.predict(X_arr), dtype=np.float64))
		ref_pred = ensure_2d(np.asarray(reference_model.predict(X_arr), dtype=np.float64))

		base_weights = self._extract_output_feature_weights(base_model, n_outputs=n_outputs, n_features=n_features)
		ref_weights = self._extract_output_feature_weights(reference_model, n_outputs=n_outputs, n_features=n_features)

		self.base_snapshot = ModelSnapshot(predictions=base_pred, weights=base_weights)
		self.reference_snapshots[cause] = ModelSnapshot(predictions=ref_pred, weights=ref_weights)

		start = int(col_offsets[cause_index])
		end = int(col_offsets[cause_index + 1])
		lag_order = max(end - start, 1)

		base_error, ref_error, f_values, p_values = error_and_p_values(
			y_true=y_true,
			y_base_pred=base_pred,
			y_ref_pred=ref_pred,
			lag_order=lag_order,
			n_features=n_features,
		)

		self.matrices.base_error.set_column(cause, base_error)
		self.matrices.ref_error.set_column(cause, ref_error)
		self.matrices.f_test.set_column(cause, f_values)
		self.matrices.p_value.set_column(cause, p_values)

		sign_values = self._sign_from_block(base_weights[:, start:end])
		self.matrices.sign.set_column(cause, sign_values)

	def result(self, threshold: float = 0.01, with_sign: bool = False) -> pd.DataFrame:
		"""Return binary/signed causality matrix derived from p-value table."""
		return self.matrices.result(threshold=threshold, with_sign=with_sign)

	@property
	def base_error(self) -> pd.DataFrame:
		return self.matrices.base_error.data

	@property
	def ref_error(self) -> pd.DataFrame:
		return self.matrices.ref_error.data

	@property
	def F_test(self) -> pd.DataFrame:
		return self.matrices.f_test.data

	@property
	def p_value(self) -> pd.DataFrame:
		return self.matrices.p_value.data

	@property
	def sign(self) -> pd.DataFrame:
		return self.matrices.sign.data

	@property
	def base_weights(self) -> Optional[NDArray[np.float64]]:
		return None if self.base_snapshot is None else self.base_snapshot.weights

	@property
	def base_predictions(self) -> Optional[NDArray[np.float64]]:
		return None if self.base_snapshot is None else self.base_snapshot.predictions

	@property
	def ref_weights(self) -> Dict[str, NDArray[np.float64]]:
		return {k: v.weights for k, v in self.reference_snapshots.items()}

	@property
	def ref_predictions(self) -> Dict[str, NDArray[np.float64]]:
		return {k: v.predictions for k, v in self.reference_snapshots.items()}
