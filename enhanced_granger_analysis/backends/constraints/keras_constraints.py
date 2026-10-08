import importlib
from importlib.util import find_spec
from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .base_constaint import process_user_relations
from ...core.constraints_config import ProcessedConstraintSpec, RelationMap
from ...core.exceptions import BackendNotAvailableError, ConstraintConfigurationError


if find_spec("keras") is not None:
    keras = importlib.import_module("keras")
else:  # pragma: no cover - runtime dependency check
    keras = None

def _ensure_keras() -> None:
    if keras is None:
        raise BackendNotAvailableError("Keras is required to use constraints.")


@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="KerasZeroMaskConstraint"
)
class KerasZeroMaskConstraint(keras.constraints.Constraint):
    def __init__(self, mask):
        _ensure_tensorflow()
        mask_arr = keras.ops.array(mask, dtype="float32")
        self.mask = mask_arr
        self._mask_t = keras.ops.transpose(mask_arr)

    def __call__(self, w):
        return w * keras.ops.cast(self._mask_t, w.dtype)

    def get_config(self):
        return {"mask": keras.ops.convert_to_numpy(self.mask).tolist()}


@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="KerasMaskAndMinAbsSumConstraint",
)
class KerasMaskAndMinAbsSumConstraint(keras.constraints.Constraint):
    """Hard-mask Dense coefficients and enforce a minimum L1 sum on disjoint groups.

    spec.mask: (n_outputs, n_features)
    Dense kernel: (n_features, n_outputs)

    IMPORTANT:
    Every coefficient addressed by a min-abs-sum rule must be unique as a
    pair ``(feature_index, output_index)``. Different groups may use the same
    feature for different outputs, or different features for the same output.
    """

    def __init__(self, spec: ProcessedConstraintSpec, eps: float = 1e-8):

        _ensure_keras()

        if not np.isfinite(eps) or eps <= 0:
            raise ConstraintConfigurationError("eps must be finite and > 0")

        mask = np.asarray(spec.mask, dtype="float32")
        if mask.ndim != 2:
            raise ConstraintConfigurationError(
                "spec.mask must be 2D with shape (n_outputs, n_features)"
            )
        if not np.all(np.isfinite(mask)):
            raise ConstraintConfigurationError("spec.mask must contain finite values")
        if not np.all((mask == 0.0) | (mask == 1.0)):
            raise ConstraintConfigurationError("spec.mask must be binary (0 or 1)")

        self.mask = mask
        self.eps = float(eps)
        self._n_outputs = int(mask.shape[0])
        self._n_features = int(mask.shape[1])
        self._kernel_shape = (self._n_features, self._n_outputs)

        # Dense kernel layout: (n_features, n_outputs)
        self._mask_t = keras.ops.cast(mask.T, dtype=np.float64)

        scatter_indices_parts: list[np.ndarray] = []
        segment_ids_parts: list[np.ndarray] = []
        min_abs_sums: list[float] = []
        group_sizes: list[int] = []
        serialized_rules: list[dict[str, Any]] = []
        seen_coefficients: dict[tuple[int, int], int] = {}

        group_id = 0

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
                    f"Rule {source_rule_index}: repeated feature_indices: {duplicates.tolist()}."
                )

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

            if min_abs_sum == 0.0:
                continue

            feature_indices_i32 = feature_indices.astype("int32", copy=False)

            scatter_indices_parts.append(
                np.column_stack(
                    (
                        feature_indices_i32,
                        np.full(feature_indices_i32.shape, output_index, dtype="int32"),
                    )
                )
            )
            segment_ids_parts.append(
                np.full(feature_indices_i32.shape, group_id, dtype="int32")
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

            group_id += 1

        self.rules = serialized_rules
        self._n_groups = group_id

        if self._n_groups:
            self._scatter_indices = keras.ops.cast(
                np.concatenate(scatter_indices_parts, axis=0), dtype="int32"
            )
            self._segment_ids = keras.ops.cast(
                np.concatenate(segment_ids_parts, axis=0), dtype="int32"
            )
            self._group_min_abs_sums = keras.ops.cast(min_abs_sums, dtype="float32")
            self._group_sizes = keras.ops.cast(group_sizes, dtype="float32")
        else:
            self._scatter_indices = keras.ops.cast(
                np.empty((0, 2), dtype="int32"), dtype="int32"
            )
            self._segment_ids = keras.ops.cast([], dtype="int32")
            self._group_min_abs_sums = keras.ops.cast([], dtype="float32")
            self._group_sizes = keras.ops.cast([], dtype="float32")

    def __call__(self, w: Any) -> Any:
        shape = tuple(w.shape)

        if w.shape.rank != 2 or any(dim is None for dim in shape):
            raise ConstraintConfigurationError(
                f"Expected a rank-2 kernel with static shape; got {shape}."
            )

        actual_shape = tuple(int(dim) for dim in shape)
        if actual_shape != self._kernel_shape:
            raise ConstraintConfigurationError(
                f"Kernel shape {actual_shape} does not match "
                f"expected {self._kernel_shape}."
            )

        mask_t = keras.ops.cast(self._mask_t, w.dtype)
        constrained = w * mask_t

        if self._n_groups == 0:
            return constrained

        _rows, n_cols = self._kernel_shape

        indices = keras.ops.cast(self._scatter_indices, "int32")
        row_indices = indices[:, 0]
        col_indices = indices[:, 1]

        flat_indices = row_indices * n_cols + col_indices
        flat_constrained = keras.ops.reshape(constrained, (-1,))

        selected = keras.ops.take(flat_constrained, flat_indices, axis=0)
        abs_selected = keras.ops.absolute(selected)

        group_abs_sums = keras.ops.segment_sum(
            abs_selected,
            self._segment_ids,
            num_segments=self._n_groups,
            sorted=False,
        )

        minimum_sums = keras.ops.cast(self._group_min_abs_sums, w.dtype)
        group_sizes = keras.ops.cast(self._group_sizes, w.dtype)

        deficits = keras.ops.maximum(minimum_sums - group_abs_sums, 0.0)

        deltas = keras.ops.take(
            deficits / group_sizes,
            self._segment_ids,
            axis=0,
        )

        signs = keras.ops.where(
            keras.ops.equal(selected, 0),
            keras.ops.ones_like(selected),
            keras.ops.sign(selected),
        )

        projected = signs * (abs_selected + deltas)

        flat_constrained = keras.ops.scatter_update(
            flat_constrained,
            keras.ops.expand_dims(flat_indices, axis=1),
            projected,
        )

        constrained = keras.ops.reshape(flat_constrained, self._kernel_shape)

        return constrained * mask_t

    def get_config(self) -> Dict[str, Any]:
        return dict(mask=self.mask.tolist(), rules=[dict(rule) for rule in self.rules], eps=self.eps)


def build_keras_constraint_from_relations(
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
    if len(spec.rules) == 0:
        return KerasZeroMaskConstraint(spec.mask)
    else:
        return KerasMaskAndMinAbsSumConstraint(spec=spec, eps=eps)
