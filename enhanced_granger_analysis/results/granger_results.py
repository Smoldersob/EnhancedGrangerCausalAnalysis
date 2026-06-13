from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.exceptions import ResultsError
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
			raise ResultsError("Model must define get_weights()")

		weights_list = model.get_weights()
		if not isinstance(weights_list, list) or len(weights_list) == 0:
			raise ResultsError("model.get_weights() must return a non-empty list")

		kernel = np.asarray(weights_list[0], dtype=np.float64)
		if kernel.ndim != 2:
			raise ResultsError(f"Expected 2D kernel matrix, got shape {kernel.shape}")

		# Preferred convention in current models: (n_features, n_outputs)
		if kernel.shape == (n_features, n_outputs):
			return kernel.T
		# Fallback if backend returns (n_outputs, n_features)
		if kernel.shape == (n_outputs, n_features):
			return kernel

		raise ResultsError(
			"Cannot infer weight orientation from kernel shape "
			f"{kernel.shape}; expected {(n_features, n_outputs)} or {(n_outputs, n_features)}"
		)

	@staticmethod
	def _coerce_output_feature_weights(
		weights: NDArray[np.float64],
		n_outputs: int,
		n_features: int,
	) -> NDArray[np.float64]:
		"""Coerce explicit weights into shape (n_outputs, n_features)."""
		arr = np.asarray(weights, dtype=np.float64)
		if arr.ndim != 2:
			raise ResultsError(f"Expected 2D weights matrix, got shape {arr.shape}")
		if arr.shape == (n_outputs, n_features):
			return arr
		if arr.shape == (n_features, n_outputs):
			return arr.T
		raise ResultsError(
			f"Cannot infer explicit weights orientation from shape {arr.shape}; "
			f"expected {(n_outputs, n_features)} or {(n_features, n_outputs)}"
		)

	@staticmethod
	def _sign_from_block(weight_block: NDArray[np.float64]) -> NDArray[np.float64]:
		"""
		Compute sign per output using coefficient with maximum absolute value.

		For each output row, this matches requirement #sym:sign.
		"""
		if weight_block.ndim != 2:
			raise ResultsError("weight_block must be 2D")
		if weight_block.shape[1] == 0:
			return np.zeros(weight_block.shape[0], dtype=np.float64)

		argmax = np.argmax(np.abs(weight_block), axis=1)
		selected = weight_block[np.arange(weight_block.shape[0]), argmax]
		return np.sign(selected).astype(np.float64)

	def update_cause(
		self,
		cause: str,
		cause_index: int,
		y_true: NDArray[np.float64],
		col_offsets: NDArray[np.int_],
		base_predictions: Optional[NDArray[np.float64]] = None,
		reference_predictions: Optional[NDArray[np.float64]] = None,
		base_weights: Optional[NDArray[np.float64]] = None,
		reference_weights: Optional[NDArray[np.float64]] = None,
	) -> None:
		"""Update all matrices and snapshots for one tested cause variable."""
		if cause not in self.causes:
			raise ResultsError(f"Unknown cause: {cause}")
		if cause_index < 0 or cause_index + 1 >= len(col_offsets):
			raise ResultsError("cause_index out of range for provided col_offsets")

		y_true_2d = ensure_2d(np.asarray(y_true, dtype=np.float64))
		n_outputs = y_true_2d.shape[1]

		if base_predictions is None:
			raise ResultsError("Provide base_predictions")
		else:
			base_pred = ensure_2d(np.asarray(base_predictions, dtype=np.float64))

		if reference_predictions is not None:
			ref_pred = ensure_2d(np.asarray(reference_predictions, dtype=np.float64))
		else:
			raise ResultsError("Provide reference_predictions")

		if base_weights is not None:
			base_weights_arr = np.asarray(base_weights, dtype=np.float64)
			if base_weights_arr.ndim != 2:
				raise ResultsError(f"Expected 2D weights matrix, got shape {base_weights_arr.shape}")
			if base_weights_arr.shape[0] == n_outputs:
				n_features = int(base_weights_arr.shape[1])
			elif base_weights_arr.shape[1] == n_outputs:
				n_features = int(base_weights_arr.shape[0])
			else:
				raise ResultsError(
					f"Cannot infer explicit weights orientation from shape {base_weights_arr.shape}; "
					f"expected {(n_outputs, 'n_features')} or {('n_features', n_outputs)}"
				)
			base_w = self._coerce_output_feature_weights(base_weights_arr, n_outputs=n_outputs, n_features=n_features)
		else:
			raise ResultsError("Provide base_weights")

		if reference_weights is not None:
			reference_weights_arr = np.asarray(reference_weights, dtype=np.float64)
			if reference_weights_arr.ndim != 2:
				raise ResultsError(f"Expected 2D weights matrix, got shape {reference_weights_arr.shape}")
			if reference_weights_arr.shape[0] == n_outputs:
				n_features = int(reference_weights_arr.shape[1])
			elif reference_weights_arr.shape[1] == n_outputs:
				n_features = int(reference_weights_arr.shape[0])
			else:
				raise ResultsError(
					f"Cannot infer explicit weights orientation from shape {reference_weights_arr.shape}; "
					f"expected {(n_outputs, 'n_features')} or {('n_features', n_outputs)}"
				)
			ref_w = self._coerce_output_feature_weights(reference_weights_arr, n_outputs=n_outputs, n_features=n_features)
		else:
			raise ResultsError("Provide reference_weights")

		self.base_snapshot = ModelSnapshot(predictions=base_pred, weights=base_w)
		self.reference_snapshots[cause] = ModelSnapshot(predictions=ref_pred, weights=ref_w)

		start = int(col_offsets[cause_index])
		end = int(col_offsets[cause_index + 1])
		lag_order = max(end - start, 1)

		base_error, ref_error, f_values, p_values = error_and_p_values(
			y_true=y_true_2d,
			y_base_pred=base_pred,
			y_ref_pred=ref_pred,
			lag_order=lag_order,
			n_features=n_features,
		)

		self.matrices.base_error.set_column(cause, base_error)
		self.matrices.ref_error.set_column(cause, ref_error)
		self.matrices.f_test.set_column(cause, f_values)
		self.matrices.p_value.set_column(cause, p_values)

		sign_values = self._sign_from_block(base_w[:, start:end])
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
