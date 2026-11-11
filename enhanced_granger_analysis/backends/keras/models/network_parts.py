from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional
from collections.abc import Sequence

from ....core.models.input_spec import ModelInputSpec
import keras


@keras.utils.register_keras_serializable(
	package="EnhancedGrangerCausalAnalysis",
	name="FeatureMask",
)
class FeatureMask(keras.layers.Layer):
	"""Input masking layer"""

	def __init__(self, n_features: int, **kwargs: Any) -> None:
		kwargs.setdefault("trainable", False)
		super().__init__(**kwargs)

		self.n_features = int(n_features)
		self.mask = self.add_weight(
			name="mask",
			shape=(self.n_features,),
			initializer="ones",
			trainable=False,
		)

	def build(self, input_shape: Any) -> None:
		super().build(input_shape)

	def call(self, inputs: Any) -> Any:
		return keras.ops.multiply(inputs, self.mask)

	def get_config(self) -> dict[str, Any]:
		config = super().get_config()
		config.update({"n_features": self.n_features})
		return config

@keras.utils.register_keras_serializable(
	package="EnhancedGrangerCausalAnalysis",
	name="LinearGrangerNetwork",
)
class LinearGrangerNetwork(keras.Model):
	"""Linear Granger model in keras for any backend."""

	def __init__(
		self,
		n_features: int,
		n_outputs: int,
		constraint: Optional[Any] = None,
		regularizer: Optional[Any] = None,
		use_bias: bool = True,
		**kwargs: Any,
	):
		kwargs.setdefault("name", "SimpleKerasGrangerModel")
		super().__init__(**kwargs)

		self.n_features = int(n_features)
		self.n_outputs = int(n_outputs)
		if not isinstance(use_bias, bool):
			raise TypeError("use_bias must be a bool.")
		self.use_bias = use_bias

		self.variable_control_layer = FeatureMask(
			n_features=self.n_features,
			name="variable_control",
		)

		self.coefficient_layer = keras.layers.Dense(
			units=self.n_outputs,
			use_bias=self.use_bias,
			kernel_constraint=constraint,
			kernel_regularizer=regularizer,
			name="coefficients",
		)
		

	def build(self, input_shape: Any) -> None:
		input_shape = tuple(input_shape)

		if len(input_shape) != 2:
			raise ValueError(
				"Expected input shape (n_samples, n_features)."
			)

		if input_shape[-1] != self.n_features:
			raise ValueError(
				f"Expected {self.n_features} input features, "
				f"received shape {input_shape}."
			)

		if not self.variable_control_layer.built:
			self.variable_control_layer.build(input_shape)

		# FeatureMask preserves the input shape.
		if not self.coefficient_layer.built:
			self.coefficient_layer.build(input_shape)

		super().build(input_shape)

	def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
		chosen = self.variable_control_layer(inputs)
		return self.coefficient_layer(chosen, training=training)

	def get_config(self) -> dict[str, Any]:
		config = super().get_config()
		config.update(
			{
				"n_features": self.n_features,
				"n_outputs": self.n_outputs,
				"use_bias": self.use_bias,
				"constraint": keras.saving.serialize_keras_object(
					self.coefficient_layer.kernel_constraint
				),
				"regularizer": keras.saving.serialize_keras_object(
					self.coefficient_layer.kernel_regularizer
				),
			}
		)
		return config

	@classmethod
	def from_config(
			cls,
			config: Dict[str, Any],
	) -> "LinearGrangerNetwork":
		config = dict(config)

		for key in ("constraint", "regularizer"):
			value = config.get(key)
			if isinstance(value, dict):
				config[key] = keras.saving.deserialize_keras_object(value)

		return cls(**config)


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


@keras.utils.register_keras_serializable(
    package="EnhancedGrangerCausalAnalysis",
    name="ClassicalLinearGrangerNetwork",
)
class ClassicalLinearGrangerNetwork(LinearGrangerNetwork):
    """Multi-output predictor with a shared autoregressive baseline.

    Uses one Dense kernel with shape:
        (n_features, n_outputs)

    Prediction:
        X @ (kernel * A.T)
        + (X * control) @ (kernel * E.T)
        + bias

    A and E are fixed complementary selectors.
    Only the feature-control vector changes between Granger tests.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        *,
        ar_selector: Any,
        constraint: Optional[Any] = None,
        regularizer: Optional[Any] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        selector = np.asarray(ar_selector)

        expected_shape = (int(n_outputs), int(n_features))

        if selector.shape != expected_shape:
            raise ValueError(
                f"Expected ar_selector shape {expected_shape}, "
                f"got {selector.shape}."
            )

        if selector.dtype.kind not in "biuf":
            raise TypeError(
                "ar_selector must contain boolean or numeric values."
            )

        if not np.all((selector == 0) | (selector == 1)):
            raise ValueError(
                "ar_selector must contain only zeros and ones."
            )

        selector = np.array(selector, dtype=bool, copy=True)

        if not np.all(selector.any(axis=1)):
            raise ValueError(
                "Every output must have at least one "
                "autoregressive feature."
            )

        kwargs.setdefault("name", "ClassicalKerasGrangerModel")

        super().__init__(
            n_features=n_features,
            n_outputs=n_outputs,
            constraint=constraint,
            regularizer=regularizer,
            use_bias=use_bias,
            **kwargs,
        )

        # Immutable architecture configuration, not Keras weights.
        self._ar_selector_config = tuple(
            tuple(bool(value) for value in row)
            for row in selector
        )

        self.reset_to_base()

    def call(
        self,
        inputs: Any,
        training: Optional[bool] = None,
    ) -> Any:
        kernel = self.coefficient_layer.kernel

        # Constants have kernel orientation: (n_features, n_outputs).
        ar_selector = keras.ops.convert_to_tensor(
            self._ar_selector_config,
            dtype=kernel.dtype,
        )
        ar_selector = keras.ops.transpose(ar_selector)
        external_selector = (
            keras.ops.ones_like(ar_selector) - ar_selector
        )

        autoregressive = keras.ops.matmul(
            inputs,
            kernel * ar_selector,
        )

        controlled_inputs = self.variable_control_layer(inputs)

        additional = keras.ops.matmul(
            controlled_inputs,
            kernel * external_selector,
        )

        prediction = autoregressive + additional

        if self.coefficient_layer.bias is not None:
            prediction = prediction + self.coefficient_layer.bias

        return prediction

    def reset_to_base(self) -> None:
        """Disable additional inputs without changing predictor weights."""
        self.variable_control_layer.mask.assign(
            np.zeros(self.n_features, dtype=np.float32)
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

        mask = np.zeros(self.n_features, dtype=np.float32)
        mask[indices] = 1.0

        self.variable_control_layer.mask.assign(mask)

    def get_controlled_slope_mask(self) -> np.ndarray:
        """Return architectural activity before structural constraints."""
        ar_selector = np.asarray(
            self._ar_selector_config,
            dtype=bool,
        )
        control = np.asarray(
            keras.ops.convert_to_numpy(
                self.variable_control_layer.mask
            ),
            dtype=bool,
        )

        return np.array(
            ar_selector
            | ((~ar_selector) & control[None, :]),
            dtype=bool,
            copy=True,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "ar_selector": [
                    list(row)
                    for row in self._ar_selector_config
                ],
            }
        )
        return config
