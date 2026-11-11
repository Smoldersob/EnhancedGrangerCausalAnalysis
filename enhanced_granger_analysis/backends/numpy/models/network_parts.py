from __future__ import annotations

from typing import Any

import numpy as np
from collections.abc import Sequence

from ....core.models.input_spec import ModelInputSpec

from ..constraints.base import BaseNumpyConstraint
from ..regularizers.base import BaseNumpyRegularizer


def positive_int(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def real_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)

    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values.")

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float64, copy=True)

    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float64 values.")

    return result


class NumpyFeatureMask:
    """Input control independent of predictor parameters."""

    trainable = False

    def __init__(self, n_features: int) -> None:
        self.n_features = positive_int(n_features, "n_features")
        self._mask = np.ones(self.n_features, dtype=np.float64)

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    def set_mask(self, value: Any) -> None:
        raw = np.asarray(value)

        if raw.dtype.kind not in "bifu":
            raise TypeError("Mask must be boolean or real numeric.")

        if raw.shape != (self.n_features,):
            raise ValueError(
                f"Expected mask shape {(self.n_features,)}."
            )

        if (
            not np.isfinite(raw).all()
            or not ((raw == 0) | (raw == 1)).all()
        ):
            raise ValueError(
                "Mask must contain only finite zeros and ones."
            )

        self._mask = np.array(raw, dtype=np.float64, copy=True)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs * self._mask

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        return self(inputs)


class NumpyLinear:
    """Multi-output layer with native (outputs, features) weights."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        use_bias: bool,
    ) -> None:
        self.weight = np.zeros(
            (n_outputs, n_features),
            dtype=np.float64,
        )
        self.bias = (
            np.zeros(n_outputs, dtype=np.float64)
            if use_bias
            else None
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        result = inputs @ self.weight.T

        if self.bias is not None:
            result = result + self.bias

        return result


class LinearGrangerNetwork:
    """Predictor architecture without training data or lifecycle state.

    Native coefficients: (n_outputs, n_features).
    Exported kernel:     (n_features, n_outputs).

    Input omission affects the forward path only. Regularization and
    structural constraints operate on raw coefficient parameters.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        constraint: BaseNumpyConstraint | None = None,
        regularizer: BaseNumpyRegularizer | None = None,
        use_bias: bool = True,
    ) -> None:
        self.n_features = positive_int(n_features, "n_features")
        self.n_outputs = positive_int(n_outputs, "n_outputs")

        if not isinstance(use_bias, bool):
            raise TypeError("use_bias must be a bool.")

        if constraint is not None and not isinstance(
            constraint, BaseNumpyConstraint
        ):
            raise TypeError(
                "constraint must be a BaseNumpyConstraint or None."
            )

        if regularizer is not None and not isinstance(
            regularizer, BaseNumpyRegularizer
        ):
            raise TypeError(
                "regularizer must be a BaseNumpyRegularizer or None."
            )

        shape = (self.n_outputs, self.n_features)

        if constraint is not None and constraint.mask.shape != shape:
            raise ValueError(
                f"Constraint must have coefficient shape {shape}."
            )

        if (
            regularizer is not None
            and regularizer.data.layout.coefficient_shape != shape
        ):
            raise ValueError(
                f"Regularizer must have coefficient shape {shape}."
            )

        self.use_bias = use_bias
        self.constraint = constraint
        self.regularizer = regularizer

        self.variable_control_layer = NumpyFeatureMask(
            self.n_features
        )
        self.coefficient_layer = NumpyLinear(
            self.n_features,
            self.n_outputs,
            self.use_bias,
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features:
            raise ValueError(
                "Expected inputs with shape "
                f"(n_samples, {self.n_features})."
            )

        masked = self.variable_control_layer(inputs)
        return self.coefficient_layer(masked)

    def coefficient_array(
        self,
        value: Any,
        name: str,
    ) -> np.ndarray:
        result = real_array(value, name)
        expected = (self.n_outputs, self.n_features)

        if result.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}."
            )

        return result

    def regularization_value_and_grad(
        self,
    ) -> tuple[float, np.ndarray]:
        weights = self.coefficient_layer.weight

        if self.regularizer is None:
            return 0.0, np.zeros_like(weights)

        penalty, gradient = self.regularizer.value_and_grad(weights)
        return float(penalty), gradient

    def regularization_loss(self) -> float:
        penalty, _ = self.regularization_value_and_grad()
        return penalty

    def constrain(self, weights: np.ndarray) -> np.ndarray:
        result = self.coefficient_array(weights, "Coefficients")

        if self.constraint is not None:
            result = self.coefficient_array(
                self.constraint(result),
                "Constraint result",
            )

            if not self.constraint.is_satisfied(result):
                raise ValueError(
                    "Constraint enforcement returned "
                    "unsatisfied coefficients."
                )

        return result

    def apply_constraints(self) -> None:
        layer = self.coefficient_layer
        layer.weight[...] = self.constrain(layer.weight)

    def get_predictor_weights(self) -> list[np.ndarray]:
        layer = self.coefficient_layer
        result = [layer.weight.T.copy()]

        if layer.bias is not None:
            result.append(layer.bias.copy())

        return result

    def set_predictor_weights(
        self,
        weights: list[np.ndarray],
    ) -> None:
        shapes = [(self.n_features, self.n_outputs)]

        if self.use_bias:
            shapes.append((self.n_outputs,))

        if not isinstance(weights, list) or len(weights) != len(shapes):
            raise ValueError(
                f"Expected a list of {len(shapes)} parameter arrays."
            )

        arrays = []

        for index, (value, shape) in enumerate(zip(weights, shapes)):
            array = real_array(value, f"Parameter {index}")

            if array.shape != shape:
                raise ValueError(
                    f"Parameter {index}: expected shape {shape}, "
                    f"got {array.shape}."
                )

            arrays.append(array)

        # Validation is complete before any parameter is changed.
        layer = self.coefficient_layer
        layer.weight[...] = arrays[0].T

        if layer.bias is not None:
            layer.bias[...] = arrays[1]


### Classical Linear Granger Network with shared AR baseline

def validate_feature_indices(
    variable_indices: Sequence[int],
    *,
    n_features: int,
) -> np.ndarray:
    """Validate column indices of the already prepared, lagged X."""
    indices = list(variable_indices)

    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise TypeError(
            "Feature indices must be integers, not booleans."
        )

    if any(index < 0 or index >= n_features for index in indices):
        raise ValueError("Feature index is out of range.")

    return np.asarray(indices, dtype=np.intp)


def build_ar_selector(input_spec: ModelInputSpec) -> np.ndarray:
    """Select positive-lag own-history features for each output."""
    if not isinstance(input_spec, ModelInputSpec):
        raise TypeError("input_spec must be a ModelInputSpec.")

    selector = np.zeros(
        (input_spec.n_outputs, input_spec.n_features),
        dtype=bool,
    )
    lags = np.asarray(input_spec.feature_lags, dtype=np.int64)

    for output_index, effect in enumerate(input_spec.effects):
        block = input_spec.cause_slices.get(effect)

        if block is None:
            raise ValueError(
                f"Output {effect!r} has no own-history block in X. "
                "The classical restricted baseline requires its "
                "lagged values among the prepared predictors."
            )

        columns = np.arange(
            block.start,
            block.stop,
            dtype=np.intp,
        )
        own_history = columns[lags[columns] > 0]

        if own_history.size == 0:
            raise ValueError(
                f"Output {effect!r} has no positive-lag "
                "own-history features in X."
            )

        selector[output_index, own_history] = True

    return selector


class ClassicalLinearGrangerNetwork(LinearGrangerNetwork):
    """Shared multi-output AR baseline with controlled extra inputs.

    Native weights:
        (n_outputs, n_features)

    Exported kernel:
        (n_features, n_outputs)

    Constraints and regularization operate on raw predictor weights.
    Dynamic feature control affects only the additional-input path.
    """

    def __init__(
        self,
        *,
        input_spec: ModelInputSpec,
        constraint: BaseNumpyConstraint | None = None,
        regularizer: BaseNumpyRegularizer | None = None,
        use_bias: bool = True,
    ) -> None:
        ar_selector = build_ar_selector(input_spec)

        super().__init__(
            n_features=input_spec.n_features,
            n_outputs=input_spec.n_outputs,
            constraint=constraint,
            regularizer=regularizer,
            use_bias=use_bias,
        )

        self._ar_selector = ar_selector.copy()
        self._external_selector = ~ar_selector

        self.reset_to_base()

    def get_controlled_slope_mask(self) -> np.ndarray:
        """Architectural activity, before structural constraints."""
        control = self.variable_control_layer.mask.astype(bool)

        return np.array(
            self._ar_selector
            | (self._external_selector & control[None, :]),
            dtype=bool,
            copy=True,
        )

    def effective_weight(self) -> np.ndarray:
        """Coefficients used by prediction, without modifying parameters."""
        return (
            self.coefficient_layer.weight
            * self.get_controlled_slope_mask()
        )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features:
            raise ValueError(
                "Expected inputs with shape "
                f"(n_samples, {self.n_features})."
            )

        result = inputs @ self.effective_weight().T

        if self.coefficient_layer.bias is not None:
            result = result + self.coefficient_layer.bias

        return result

    def reset_to_base(self) -> None:
        """Disable additional inputs, preserving predictor parameters."""
        self.variable_control_layer.set_mask(
            np.zeros(self.n_features, dtype=np.float64)
        )

    def select_additional_features(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Replace selected additional inputs; never accumulate."""
        indices = validate_feature_indices(
            variable_indices,
            n_features=self.n_features,
        )

        mask = np.zeros(self.n_features, dtype=np.float64)
        mask[indices] = 1.0

        self.variable_control_layer.set_mask(mask)