from __future__ import annotations
from typing import Any, Dict

import numpy as np
import torch

from .base import BasePyTorchConstraint
from .registry import register_constraint
from ....core.components.constraint_spec import PreparedConstraintData
from ....core.exceptions import ConstraintConfigurationError

@register_constraint("MaskConstraint", "mask_constraint", "mask")
class MaskConstraint(BasePyTorchConstraint):
	"""Hard mask constraint for PyTorch coefficient matrices.

	Expected weight shape: (n_outputs, n_features) for nn.Linear(in, out).weight.
	Expected mask shape: (n_outputs, n_features).
	"""
	def __init__(self, data: PreparedConstraintData):
		super().__init__(data)

		if self.rules:
			raise ConstraintConfigurationError(
				"MaskConstraint cannot enforce min_abs_sum rules. "
				"Use mask_and_min_abs_sum."
			)

	def enforce(self, params: Any) -> Any:
		if not torch.is_tensor(params):
			raise ConstraintConfigurationError("params must be torch.Tensor")
		if tuple(params.shape) != tuple(self.mask.shape):
			raise ConstraintConfigurationError(
				f"params shape {tuple(params.shape)} does not match mask shape {self.mask.shape}"
			)
		mask_t = torch.as_tensor(self.mask, dtype=params.dtype, device=params.device)
		return params * mask_t

	def is_satisfied(self, params: Any) -> bool:
		if not torch.is_tensor(params) or tuple(params.shape) != tuple(self.mask.shape):
			return False
		if not bool(torch.isfinite(params).all().item()):
			return False
		mask = torch.as_tensor(self.mask, dtype=params.dtype, device=params.device)
		return bool(torch.all(torch.abs(params * (1.0 - mask)) <= 1e-8).item())


@register_constraint("MaskAndMinAbsSumConstraint", "mask_and_min_abs_sum_constraint", "mask_and_min_abs_sum")
class MaskAndMinAbsSumConstraint(BasePyTorchConstraint):
	"""Mask + min abs-sum constraint for PyTorch coefficient matrices."""

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
		
	def get_config(self) -> Dict[str, Any]:
		return {**super().get_config(), "eps": self.eps}

	def enforce(self, params: Any) -> Any:
		if not torch.is_tensor(params):
			raise ConstraintConfigurationError("params must be torch.Tensor")
		if tuple(params.shape) != tuple(self.mask.shape):
			raise ConstraintConfigurationError(
				f"params shape {tuple(params.shape)} does not match mask shape {self.mask.shape}"
			)

		mask_t = torch.as_tensor(self.mask, dtype=params.dtype, device=params.device)
		constrained = params * mask_t

		for rule in self.rules:
			out_idx = int(rule.output_index)
			feat_idx = list(rule.feature_indices)
			if len(feat_idx) == 0:
				continue

			idx_t = torch.as_tensor(feat_idx, dtype=torch.long, device=params.device)
			selected = constrained[out_idx, idx_t]
			current_sum = torch.sum(torch.abs(selected))
			min_sum = torch.tensor(float(rule.min_abs_sum), dtype=params.dtype, device=params.device)
			deficit = torch.clamp(min_sum - current_sum, min=0.0)
			if float(deficit.item()) == 0.0:
				continue

			n_sel = float(len(feat_idx))
			delta = deficit / n_sel
			sign = torch.sign(selected)
			sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
			updated = sign * (torch.abs(selected) + delta)
			constrained[out_idx, idx_t] = updated

		# Re-apply hard mask after all rule updates.
		constrained = constrained * mask_t
		return constrained

	def is_satisfied(self, params: Any) -> bool:
		if not torch.is_tensor(params) or tuple(params.shape) != tuple(self.mask.shape):
			return False
		if not bool(torch.isfinite(params).all().item()):
			return False
		mask = torch.as_tensor(self.mask, dtype=params.dtype, device=params.device)
		if not bool(torch.all(torch.abs(params * (1.0 - mask)) <= self.eps).item()):
			return False
		for rule in self.rules:
			indices = torch.as_tensor(rule.feature_indices, dtype=torch.long, device=params.device)
			total = torch.sum(torch.abs(params[rule.output_index, indices]))
			if float(total.item()) + self.eps < rule.min_abs_sum:
				return False
		return True


