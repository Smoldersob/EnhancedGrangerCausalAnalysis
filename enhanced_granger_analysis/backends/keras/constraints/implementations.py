from typing import Any

import keras
import numpy as np

from .base import BaseKerasConstraint
from .registry import register_constraint
from ....core.components.constraint_spec import PreparedConstraintData
from ....core.exceptions import ConstraintConfigurationError



@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="MaskConstraint",
)
@register_constraint("MaskConstraint", "mask_constraint", "mask")
class MaskConstraint(BaseKerasConstraint):
    def __init__(self, data: PreparedConstraintData):
        super().__init__(data)

        if self.rules:
            raise ConstraintConfigurationError(
                "MaskConstraint cannot enforce min_abs_sum rules. "
                "Use mask_and_min_abs_sum."
            )

    def enforce(self, params):
        shape = tuple(params.shape)

        if shape != self._kernel_shape:
            raise ConstraintConfigurationError(
                f"Kernel shape {shape} does not match "
                f"{self._kernel_shape}"
            )

        mask_t = keras.ops.cast(self._mask_t, params.dtype)
        return params * mask_t

    def is_satisfied(self, params):
        values = np.asarray(
            keras.ops.convert_to_numpy(params)
        )

        if values.shape != self._kernel_shape:
            return False

        if not np.all(np.isfinite(values)):
            return False

        return bool(np.allclose(
            values * (1.0 - self.mask.T),
            0.0,
        ))


@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="MaskAndMinAbsSumConstraint",
)
@register_constraint("MaskAndMinAbsSumConstraint", "mask_and_min_abs_sum_constraint", "mask_and_min_abs_sum")
class MaskAndMinAbsSumConstraint(BaseKerasConstraint):
    """
    Hard-mask Dense coefficients and enforce a minimum L1 sum on disjoint groups.

    spec.mask: (n_outputs, n_features)
    Dense kernel: (n_features, n_outputs)

    IMPORTANT:
    Every coefficient addressed by a min-abs-sum rule must be unique as a
    pair ``(feature_index, output_index)``. Different groups may use the same
    feature for different outputs, or different features for the same output.
    """
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
        self._n_outputs = len(self._data.output_names)
        self._n_features = self._data.n_features    

        scatter_indices_parts: list[np.ndarray] = []
        segment_ids_parts: list[np.ndarray] = []
        min_abs_sums: list[float] = []
        group_sizes: list[int] = []
        seen_coefficients: dict[tuple[int, int], int] = {}

        group_id = 0

        for source_rule_index, rule in enumerate(self.rules):
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

            forbidden = feature_indices[self.mask[output_index, feature_indices] == 0.0]
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

            group_id += 1
            
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

    def enforce(self, w: Any) -> Any:
        shape = tuple(w.shape)

        if len(shape) != 2 or any(dim is None for dim in shape):
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

    def is_satisfied(self, params):
        values = np.asarray(keras.ops.convert_to_numpy(params))
        if values.shape != self._kernel_shape or not np.isfinite(values).all():
            return False
        if not np.allclose(values * (1.0 - self.mask.T), 0.0, atol=self.eps, rtol=0.0):
            return False
        return all(
            np.sum(np.abs(values[list(rule.feature_indices), rule.output_index])) + self.eps
            >= rule.min_abs_sum for rule in self.rules
        )

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}
