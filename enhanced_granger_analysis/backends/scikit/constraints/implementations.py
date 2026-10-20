from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import ScikitConstraint
from .registry import register_constraint 
from ....core.components.constraint_spec import PreparedConstraintData
from ....core.exceptions import ConstraintConfigurationError


@register_constraint("MaskConstraint", "mask_constraint", "mask")
class MaskConstraint(ScikitConstraint):
	"""Hard mask constraint for numpy/scikit coefficient matrices.

	Expected coefficient shape: (n_outputs, n_features).
	Expected mask shape: (n_outputs, n_features).
	"""
	def __init__(self, data: PreparedConstraintData):
		super().__init__(data)

		if self.rules:
			raise ConstraintConfigurationError(
				"MaskConstraint cannot enforce min_abs_sum rules. "
				"Use mask_and_min_abs_sum."
			)

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

@register_constraint("MaskAndMinAbsSumConstraint","mask_and_min_abs_sum_constraint","mask_and_min_abs_sum")
class MaskAndMinAbsSumConstraint(ScikitConstraint):
	"""Mask + min abs-sum constraint for numpy/scikit coefficient matrices."""
	def __init__(
		self,
		data: PreparedConstraintData,
		eps: float = 1e-8,
	):
		super().__init__(data)

		if not np.isfinite(eps) or eps <= 0:
			raise ConstraintConfigurationError(
				"eps must be finite and > 0"
			)

		self.eps = float(eps)

	def get_config(self):
			return {
				**super().get_config(),
				"eps": self.eps,
			}

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