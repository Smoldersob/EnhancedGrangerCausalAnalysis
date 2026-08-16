from __future__ import annotations

import importlib
from importlib.util import find_spec
from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_constaint import process_user_relations
from ...core.constraints_config import ProcessedConstraintSpec, RelationMap
from ...core.exceptions import BackendNotAvailableError, ConstraintConfigurationError


if find_spec("tensorflow") is not None:
	tf = importlib.import_module("tensorflow")
else:  # pragma: no cover - runtime dependency check
	tf = None


def _ensure_tensorflow() -> None:
	if tf is None:
		raise BackendNotAvailableError("TensorFlow is required to use tensorflow constraints.")

@tf.keras.utils.register_keras_serializable(package="EnhancedGrangerCausalAnalysis", name="TensorFlowMaskConstraint")
class TensorFlowMaskConstraint(tf.keras.constraints.Constraint if tf is not None else object):
	"""Keras kernel constraint that hard-zeros coefficients using a binary mask.

	The expected mask shape is (n_outputs, n_features), aligned with LagEngine.
	Dense kernel shape is (n_features, n_outputs), so the mask is transposed internally.
	"""

	def __init__(self, mask: NDArray[np.float32]) -> None:
		_ensure_tensorflow()

		mask_arr = np.asarray(mask, dtype=np.float32)
		if mask_arr.ndim != 2:
			raise ConstraintConfigurationError(
				"mask must be 2D with shape (n_outputs, n_features)"
			)

		self.mask = mask_arr
		self._expected_kernel_shape = (
			int(mask_arr.shape[1]),
			int(mask_arr.shape[0]),
		)
		self._mask_t = tf.constant(mask_arr.T, dtype=tf.float32)

	def __call__(self, w: Any) -> Any:
		return w * tf.cast(self._mask_t, w.dtype)

	def enforce(self, w: Any) -> Any:
		return self(w)

	def is_satisfied(self, params: Any, eps: float = 1e-8) -> bool:
		if eps <= 0 or params.shape.rank != 2:
			return False

		actual_shape = tuple(int(v) for v in params.shape)
		if actual_shape != self._expected_kernel_shape:
			return False

		mask_t = tf.cast(self._mask_t, params.dtype)
		violation = tf.abs(params * (1.0 - mask_t))

		return bool(
			tf.reduce_all(
				violation <= tf.cast(eps, params.dtype)
			).numpy()
		)

	def get_config(self) -> Dict[str, Any]:
		return {"mask": self.mask.tolist()}


@tf.keras.utils.register_keras_serializable(package="EnhancedGrangerCausalAnalysis", name="TensorFlowMaskAndMinAbsSumConstraint")
class TensorFlowMaskAndMinAbsSumConstraint(
    tf.keras.constraints.Constraint if tf is not None else object
):
    """Hard-mask Dense coefficients and enforce a minimum L1 sum on disjoint groups.

    ``spec.mask`` uses LagEngine layout ``(n_outputs, n_features)`` while a
    Dense kernel uses ``(n_features, n_outputs)``. All validation and static
    TensorFlow constants are prepared once in ``__init__``.

    Every coefficient addressed by a min-abs-sum rule must be unique as a
    pair ``(feature_index, output_index)``. Different groups may use the same
    feature for different outputs, or different features for the same output.
    """

    def __init__(self, spec: ProcessedConstraintSpec, eps: float = 1e-8) -> None:
        if tf is None:
            raise BackendNotAvailableError(
                "TensorFlow is required to use tensorflow constraints."
            )
        if not np.isfinite(eps) or eps <= 0:
            raise ConstraintConfigurationError("eps must be finite and > 0")

        mask = np.asarray(spec.mask, dtype=np.float32)
        if mask.ndim != 2:
            raise ConstraintConfigurationError(
                "spec.mask must be 2D with shape (n_outputs, n_features)"
            )
        if not np.all(np.isfinite(mask)):
            raise ConstraintConfigurationError("spec.mask must contain finite values")
        if not np.all((mask == 0.0) | (mask == 1.0)):
            raise ConstraintConfigurationError("spec.mask must be binary (values 0 or 1)")

        self.mask = mask
        self.eps = float(eps)
        self._n_outputs = int(mask.shape[0])
        self._n_features = int(mask.shape[1])
        self._kernel_shape = (self._n_features, self._n_outputs)

        # Layout already matches Dense.kernel: (n_features, n_outputs).
        self._mask_t = tf.constant(mask.T, dtype=tf.float32)

        index_parts: list[np.ndarray] = []
        segment_parts: list[np.ndarray] = []
        min_abs_sums: list[float] = []
        group_sizes: list[int] = []
        serialized_rules: list[Dict[str, Any]] = []
        seen_coefficients: dict[tuple[int, int], int] = {}

        for source_rule_index, rule in enumerate(spec.rules):
            output_index = int(rule.output_index)
            feature_indices = np.asarray(rule.feature_indices, dtype=np.int64)
            min_abs_sum = float(rule.min_abs_sum)

            if output_index < 0 or output_index >= self._n_outputs:
                raise ConstraintConfigurationError(
                    f"Rule {source_rule_index}: output_index={output_index} is outside "
                    f"[0, {self._n_outputs - 1}]."
                )
            if feature_indices.ndim != 1 or feature_indices.size == 0:
                raise ConstraintConfigurationError(
                    f"Rule {source_rule_index}: feature_indices must be a non-empty 1D sequence."
                )
            if np.any(feature_indices < 0) or np.any(feature_indices >= self._n_features):
                raise ConstraintConfigurationError(
                    f"Rule {source_rule_index}: feature_indices contain values outside "
                    f"[0, {self._n_features - 1}]."
                )
            if not np.isfinite(min_abs_sum) or min_abs_sum < 0:
                raise ConstraintConfigurationError(
                    f"Rule {source_rule_index}: min_abs_sum must be finite and >= 0."
                )

            unique_features, feature_counts = np.unique(
                feature_indices, return_counts=True
            )
            duplicates = unique_features[feature_counts > 1]
            if duplicates.size:
                raise ConstraintConfigurationError(
                    f"Rule {source_rule_index}: repeated feature_indices: "
                    f"{duplicates.tolist()}."
                )

            # A coefficient forced to zero by the hard mask cannot be part of
            # a positive lower-bound group. Reject it even for a zero threshold:
            # this catches contradictory or redundant user specifications early.
            forbidden = feature_indices[mask[output_index, feature_indices] == 0.0]
            if forbidden.size:
                raise ConstraintConfigurationError(
                    f"Rule {source_rule_index}: coefficients are hard-masked to zero "
                    f"for output_index={output_index}, feature_indices={forbidden.tolist()}."
                )

            for feature_index in feature_indices:
                coefficient = (int(feature_index), output_index)
                previous_rule_index = seen_coefficients.get(coefficient)
                if previous_rule_index is not None:
                    raise ConstraintConfigurationError(
                        "Overlapping min_abs_sum groups are not supported: "
                        f"coefficient (feature_index={coefficient[0]}, "
                        f"output_index={coefficient[1]}) appears in rules "
                        f"{previous_rule_index} and {source_rule_index}."
                    )
                seen_coefficients[coefficient] = source_rule_index

            # The constraint sum(abs(w)) >= 0 is always fulfilled. It has no
            # runtime role, but has already been fully validated above.
            if min_abs_sum == 0.0:
                continue

            feature_indices_i32 = feature_indices.astype(np.int32, copy=False)
            group_id = len(min_abs_sums)
            index_parts.append(
                np.column_stack(
                    (
                        feature_indices_i32,
                        np.full(feature_indices_i32.shape, output_index, dtype=np.int32),
                    )
                )
            )
            segment_parts.append(
                np.full(feature_indices_i32.shape, group_id, dtype=np.int32)
            )
            min_abs_sums.append(min_abs_sum)
            group_sizes.append(int(feature_indices_i32.size))
            serialized_rules.append(
                {
                    "output_index": output_index,
                    "feature_indices": feature_indices_i32.tolist(),
                    "min_abs_sum": min_abs_sum,
                }
            )

        self.rules = serialized_rules
        self._n_groups = len(min_abs_sums)

        if self._n_groups:
            self._scatter_indices = tf.constant(
                np.concatenate(index_parts, axis=0), dtype=tf.int32
            )
            self._segment_ids = tf.constant(
                np.concatenate(segment_parts, axis=0), dtype=tf.int32
            )
            self._group_min_abs_sums = tf.constant(min_abs_sums, dtype=tf.float32)
            self._group_sizes = tf.constant(group_sizes, dtype=tf.float32)
        else:
            self._scatter_indices = tf.constant(
                np.empty((0, 2), dtype=np.int32), dtype=tf.int32
            )
            self._segment_ids = tf.constant([], dtype=tf.int32)
            self._group_min_abs_sums = tf.constant([], dtype=tf.float32)
            self._group_sizes = tf.constant([], dtype=tf.float32)

    def __call__(self, w: Any) -> Any:
        """Apply the projection; this is the per-optimizer-step hot path."""
        mask_t = tf.cast(self._mask_t, w.dtype)
        constrained = w * mask_t

        if self._n_groups == 0:
            return constrained

        selected = tf.gather_nd(constrained, self._scatter_indices)
        abs_selected = tf.abs(selected)
        group_abs_sums = tf.math.unsorted_segment_sum(
            abs_selected, self._segment_ids, self._n_groups
        )
        deficits = tf.maximum(
            tf.zeros_like(group_abs_sums),
            tf.cast(self._group_min_abs_sums, w.dtype) - group_abs_sums,
        )
        deltas = tf.gather(
            deficits / tf.cast(self._group_sizes, w.dtype), self._segment_ids
        )
        signs = tf.where(tf.equal(selected, 0), tf.ones_like(selected), tf.sign(selected))
        projected = signs * (abs_selected + deltas)

        # Validated in __init__: each (feature_index, output_index) is unique,
        # and none points to a hard-masked coefficient.
        return tf.tensor_scatter_nd_update(constrained, self._scatter_indices, projected)

    def enforce(self, w: Any) -> Any:
        """Compatibility alias for callers using the old public method."""
        return self(w)

    def is_satisfied(self, params: Any) -> bool:
        """Diagnostic check; not intended for the training hot path."""
        if not tf.is_tensor(params) or params.shape.rank != 2:
            return False
        if tuple(params.shape.as_list()) != self._kernel_shape:
            return False

        mask_t = tf.cast(self._mask_t, params.dtype)
        mask_ok = tf.reduce_all(
            tf.abs(params * (1.0 - mask_t)) <= tf.cast(self.eps, params.dtype)
        )
        if not bool(mask_ok.numpy()):
            return False
        if self._n_groups == 0:
            return True

        selected = tf.gather_nd(params, self._scatter_indices)
        group_abs_sums = tf.math.unsorted_segment_sum(
            tf.abs(selected), self._segment_ids, self._n_groups
        )
        groups_ok = tf.reduce_all(
            group_abs_sums + tf.cast(self.eps, params.dtype)
            >= tf.cast(self._group_min_abs_sums, params.dtype)
        )
        return bool(groups_ok.numpy())

    def get_config(self) -> Dict[str, Any]:
        return {
            "mask": self.mask.tolist(),
            "rules": [dict(rule) for rule in self.rules],
            "eps": self.eps,
        }

def build_tensorflow_constraint_from_relations(
	relations: RelationMap,
	predictor_names: Sequence[str],
	output_names: Sequence[str],
	col_offsets: Sequence[int],
	n_features: int,
	base_mask: Optional[NDArray[np.float32]] = None,
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
