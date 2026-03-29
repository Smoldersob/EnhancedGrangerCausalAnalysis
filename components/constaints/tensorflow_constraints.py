from __future__ import annotations

import importlib
from importlib.util import find_spec
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .base_constaint import ProcessedConstraintSpec, process_user_relations


if find_spec("tensorflow") is not None:
	tf = importlib.import_module("tensorflow")
else:  # pragma: no cover - runtime dependency check
	tf = None


def _ensure_tensorflow() -> None:
	if tf is None:
		raise RuntimeError("TensorFlow is required to use tensorflow constraints.")


class TensorFlowMaskConstraint(tf.keras.constraints.Constraint if tf is not None else object):
	"""Keras kernel constraint that hard-zeros coefficients using a binary mask.

	The expected mask shape is (n_outputs, n_features), aligned with LagEngine.
	Dense kernel shape is (n_features, n_outputs), so the mask is transposed internally.
	"""

	def __init__(self, mask: NDArray[np.float64]) -> None:
		_ensure_tensorflow()
		arr = np.asarray(mask, dtype=np.float64)
		if arr.ndim != 2:
			raise ValueError("mask must be a 2D array with shape (n_outputs, n_features)")
		self.mask = arr

	def __call__(self, w: Any) -> Any:
		target_dtype = w.dtype if hasattr(w, "dtype") else tf.float32
		mask_t = tf.transpose(tf.convert_to_tensor(self.mask, dtype=target_dtype))
		return w * mask_t

	def get_config(self) -> Dict[str, Any]:
		return {"mask": self.mask.tolist()}


class TensorFlowMaskAndMinAbsSumConstraint(
	tf.keras.constraints.Constraint if tf is not None else object
):
	"""Constraint that combines hard mask zeroing and per-relation min abs-sum forcing."""

	def __init__(
		self,
		spec: ProcessedConstraintSpec,
		eps: float = 1e-8,
	) -> None:
		_ensure_tensorflow()
		self.mask = np.asarray(spec.mask, dtype=np.float64)
		if self.mask.ndim != 2:
			raise ValueError("spec.mask must be 2D with shape (n_outputs, n_features)")
		self.rules = [
			{
				"output_index": int(rule.output_index),
				"feature_indices": tuple(int(v) for v in rule.feature_indices),
				"min_abs_sum": float(rule.min_abs_sum),
			}
			for rule in spec.rules
		]
		if eps <= 0:
			raise ValueError("eps must be > 0")
		self.eps = float(eps)

	def __call__(self, w: Any) -> Any:
		target_dtype = w.dtype if hasattr(w, "dtype") else tf.float32
		mask_t = tf.transpose(tf.convert_to_tensor(self.mask, dtype=target_dtype))
		constrained = w * mask_t

		for rule in self.rules:
			out_idx = int(rule["output_index"])
			feat_idx = rule["feature_indices"]
			if len(feat_idx) == 0:
				continue

			idx_tf = tf.constant(feat_idx, dtype=tf.int32)
			col = constrained[:, out_idx]
			selected = tf.gather(col, idx_tf)

			current_sum = tf.reduce_sum(tf.abs(selected))
			min_sum = tf.cast(rule["min_abs_sum"], target_dtype)
			deficit = tf.maximum(tf.cast(0.0, target_dtype), min_sum - current_sum)
			if tf.equal(deficit, 0.0):
				continue

			n_sel = tf.cast(tf.size(selected), target_dtype)
			delta = deficit / n_sel
			sign = tf.sign(selected)
			sign = tf.where(tf.equal(sign, 0.0), tf.ones_like(sign), sign)
			updated_selected = sign * (tf.abs(selected) + delta)

			rule_col_idx = tf.fill(tf.shape(idx_tf), tf.cast(out_idx, tf.int32))
			scatter_idx = tf.stack([idx_tf, rule_col_idx], axis=1)
			constrained = tf.tensor_scatter_nd_update(constrained, scatter_idx, updated_selected)

		# Re-apply hard mask to guarantee zeroed coefficients stay zero after updates.
		constrained = constrained * mask_t
		return constrained

	def get_config(self) -> Dict[str, Any]:
		return {
			"mask": self.mask.tolist(),
			"rules": [dict(rule) for rule in self.rules],
			"eps": self.eps,
		}


def build_tensorflow_constraint_from_relations(
	relations: Mapping[Tuple[str, str], float | int | bool | str | Mapping[str, float] | None],
	predictor_names: Sequence[str],
	output_names: Sequence[str],
	col_offsets: Sequence[int],
	n_features: int,
	base_mask: Optional[NDArray[np.float64]] = None,
	eps: float = 1e-8,
) -> TensorFlowMaskAndMinAbsSumConstraint:
	"""Build TensorFlow combined constraint from user-friendly relation mapping."""
	spec = process_user_relations(
		relations=relations,
		predictor_names=predictor_names,
		output_names=output_names,
		col_offsets=col_offsets,
		n_features=n_features,
		base_mask=base_mask,
	)
	return TensorFlowMaskAndMinAbsSumConstraint(spec=spec, eps=eps)
