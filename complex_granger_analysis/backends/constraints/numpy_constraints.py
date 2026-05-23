from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_constaint import process_user_relations
from ...core.constraints_config import ProcessedConstraintSpec, RelationMap
from ...core.exceptions import ConstraintConfigurationError


class NumpyMaskConstraint:
	"""Hard mask constraint for numpy/scikit coefficient matrices.

	Expected coefficient shape: (n_outputs, n_features).
	Expected mask shape: (n_outputs, n_features).
	"""

	def __init__(self, mask: NDArray[np.float64]) -> None:
		arr = np.asarray(mask, dtype=np.float64)
		if arr.ndim != 2:
			raise ConstraintConfigurationError("mask must be 2D with shape (n_outputs, n_features)")
		self.mask = arr

	def __call__(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
		return self.enforce(params)

	def enforce(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
		w = np.asarray(params, dtype=np.float64)
		if w.shape != self.mask.shape:
			raise ConstraintConfigurationError(
				f"params shape {w.shape} does not match mask shape {self.mask.shape}"
			)
		return w * self.mask

	def is_satisfied(self, params: NDArray[np.float64]) -> bool:
		w = np.asarray(params, dtype=np.float64)
		if w.shape != self.mask.shape:
			return False
		return bool(np.allclose(w * (1.0 - self.mask), 0.0))

	def get_config(self) -> Dict[str, Any]:
		return {"mask": self.mask.tolist()}


class NumpyMaskAndMinAbsSumConstraint:
	"""Mask + min abs-sum constraint for numpy/scikit coefficient matrices."""

	def __init__(self, spec: ProcessedConstraintSpec, eps: float = 1e-8) -> None:
		mask = np.asarray(spec.mask, dtype=np.float64)
		if mask.ndim != 2:
			raise ConstraintConfigurationError("spec.mask must be 2D with shape (n_outputs, n_features)")
		if eps <= 0:
			raise ConstraintConfigurationError("eps must be > 0")

		self.mask = mask
		self.rules = tuple(spec.rules)
		self.eps = float(eps)

	def __call__(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
		return self.enforce(params)

	def enforce(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
		w = np.asarray(params, dtype=np.float64)
		if w.shape != self.mask.shape:
			raise ConstraintConfigurationError(
				f"params shape {w.shape} does not match mask shape {self.mask.shape}"
			)

		constrained = w * self.mask

		for rule in self.rules:
			out_idx = int(rule.output_index)
			feat_idx = np.asarray(rule.feature_indices, dtype=np.int64)
			if feat_idx.size == 0:
				continue

			selected = constrained[out_idx, feat_idx]
			current_sum = float(np.sum(np.abs(selected)))
			deficit = float(max(0.0, float(rule.min_abs_sum) - current_sum))
			if deficit <= 0.0:
				continue

			n_sel = float(feat_idx.size)
			delta = deficit / n_sel
			signs = np.sign(selected)
			signs[signs == 0.0] = 1.0
			constrained[out_idx, feat_idx] = signs * (np.abs(selected) + delta)

		# Re-apply hard mask after all rule updates.
		constrained = constrained * self.mask
		return constrained

	def is_satisfied(self, params: NDArray[np.float64]) -> bool:
		w = np.asarray(params, dtype=np.float64)
		if w.shape != self.mask.shape:
			return False
		if not np.allclose(w * (1.0 - self.mask), 0.0):
			return False

		for rule in self.rules:
			out_idx = int(rule.output_index)
			feat_idx = np.asarray(rule.feature_indices, dtype=np.int64)
			if feat_idx.size == 0:
				continue
			sum_abs = float(np.sum(np.abs(w[out_idx, feat_idx])))
			if sum_abs + self.eps < float(rule.min_abs_sum):
				return False
		return True

	def get_config(self) -> Dict[str, Any]:
		return {
			"mask": self.mask.tolist(),
			"rules": [
				{
					"output_index": int(rule.output_index),
					"feature_indices": list(rule.feature_indices),
					"min_abs_sum": float(rule.min_abs_sum),
				}
				for rule in self.rules
			],
			"eps": self.eps,
		}


def build_numpy_constraint_from_relations(
	relations: RelationMap,
	predictor_names: Sequence[str],
	output_names: Sequence[str],
	col_offsets: Sequence[int],
	n_features: int,
	base_mask: Optional[NDArray[np.float64]] = None,
	eps: float = 1e-8,
) -> NumpyMaskAndMinAbsSumConstraint:
	"""Build numpy/scikit combined constraint from user-friendly relation mapping."""
	spec = process_user_relations(
		relations=relations,
		predictor_names=predictor_names,
		output_names=output_names,
		col_offsets=col_offsets,
		n_features=n_features,
		base_mask=base_mask,
	)
	return NumpyMaskAndMinAbsSumConstraint(spec=spec, eps=eps)

