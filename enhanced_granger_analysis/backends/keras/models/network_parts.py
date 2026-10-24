from __future__ import annotations

from typing import Any, Dict, Optional

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
