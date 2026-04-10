from __future__ import annotations

import importlib
from importlib.util import find_spec
from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_constaint import process_user_relations
from ...core.constraints_config import ProcessedConstraintSpec, RelationMap
from ...core.exceptions import BackendNotAvailableError, ConstraintConfigurationError


if find_spec("torch") is not None:
	torch = importlib.import_module("torch")
else:  # pragma: no cover - runtime dependency check
	torch = None


def _ensure_torch() -> None:
	if torch is None:
		raise BackendNotAvailableError("PyTorch is required to use pytorch constraints.")


class PyTorchMaskConstraint:
	"""Hard mask constraint for PyTorch coefficient matrices.

	Expected weight shape: (n_outputs, n_features) for nn.Linear(in, out).weight.
	Expected mask shape: (n_outputs, n_features).
	"""

	def __init__(self, mask: NDArray[np.float64]) -> None:
		_ensure_torch()
		arr = np.asarray(mask, dtype=np.float64)
		if arr.ndim != 2:
			raise ConstraintConfigurationError("mask must be 2D with shape (n_outputs, n_features)")
		self.mask = arr

	def __call__(self, params: Any) -> Any:
		return self.enforce(params)

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
		mask_t = torch.as_tensor(self.mask, dtype=params.dtype, device=params.device)
		viol = torch.abs(params * (1.0 - mask_t))
		return bool(torch.all(viol <= 1e-12).item())

	def get_config(self) -> Dict[str, Any]:
		return {"mask": self.mask.tolist()}


class PyTorchMaskAndMinAbsSumConstraint:
	"""Mask + min abs-sum constraint for PyTorch coefficient matrices."""

	def __init__(self, spec: ProcessedConstraintSpec, eps: float = 1e-8) -> None:
		_ensure_torch()
		mask = np.asarray(spec.mask, dtype=np.float64)
		if mask.ndim != 2:
			raise ConstraintConfigurationError("spec.mask must be 2D with shape (n_outputs, n_features)")
		if eps <= 0:
			raise ConstraintConfigurationError("eps must be > 0")

		self.mask = mask
		self.rules = tuple(spec.rules)
		self.eps = float(eps)

	def __call__(self, params: Any) -> Any:
		return self.enforce(params)

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

		mask_t = torch.as_tensor(self.mask, dtype=params.dtype, device=params.device)
		viol = torch.abs(params * (1.0 - mask_t))
		if not bool(torch.all(viol <= self.eps).item()):
			return False

		for rule in self.rules:
			out_idx = int(rule.output_index)
			feat_idx = list(rule.feature_indices)
			if len(feat_idx) == 0:
				continue
			idx_t = torch.as_tensor(feat_idx, dtype=torch.long, device=params.device)
			sum_abs = torch.sum(torch.abs(params[out_idx, idx_t]))
			if float(sum_abs.item()) + self.eps < float(rule.min_abs_sum):
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


def build_pytorch_constraint_from_relations(
	relations: RelationMap,
	predictor_names: Sequence[str],
	output_names: Sequence[str],
	col_offsets: Sequence[int],
	n_features: int,
	base_mask: Optional[NDArray[np.float64]] = None,
	eps: float = 1e-8,
) -> PyTorchMaskAndMinAbsSumConstraint:
	"""Build PyTorch combined constraint from user-friendly relation mapping."""
	spec = process_user_relations(
		relations=relations,
		predictor_names=predictor_names,
		output_names=output_names,
		col_offsets=col_offsets,
		n_features=n_features,
		base_mask=base_mask,
	)
	return PyTorchMaskAndMinAbsSumConstraint(spec=spec, eps=eps)

