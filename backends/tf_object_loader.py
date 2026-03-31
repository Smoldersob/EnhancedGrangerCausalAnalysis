from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class TensorFlowObjectLoader:
	"""Resolve TensorFlow callback and optimizer specs into Keras objects."""

	def __init__(self, tf_module: Any) -> None:
		self._tf = tf_module

	@staticmethod
	def _extract_typed_spec(raw_spec: Any, *, context: str) -> Tuple[str, Dict[str, Any]]:
		if isinstance(raw_spec, str):
			return raw_spec.strip().lower(), {}

		if not isinstance(raw_spec, Mapping):
			raise ValueError(f"{context} spec must be a string or object")

		type_name = raw_spec.get("type", raw_spec.get("name", raw_spec.get("kind")))
		if not isinstance(type_name, str) or not type_name.strip():
			raise ValueError(f"{context} spec requires a non-empty 'type' field")

		params_raw = raw_spec.get("params", {})
		if params_raw is None:
			params_raw = {}
		if not isinstance(params_raw, Mapping):
			raise ValueError(f"{context} spec 'params' must be an object")

		params = dict(params_raw)
		for key, value in raw_spec.items():
			if key not in {"type", "name", "kind", "params"}:
				params[key] = value

		return type_name.strip().lower(), params

	def resolve_callback(self, raw_callback: Any) -> Any:
		keras_callback_base = self._tf.keras.callbacks.Callback
		if isinstance(raw_callback, keras_callback_base):
			return raw_callback

		type_name, params = self._extract_typed_spec(raw_callback, context="callback")

		if type_name in {"early_stopping", "keras_early_stopping", "earlystopping"}:
			return self._tf.keras.callbacks.EarlyStopping(**params)

		if type_name in {
			"reduce_lr_on_plateau",
			"reduce_learning_rate",
			"reduce_lr",
			"reducelronplateau",
			"keras_reduce_lr",
		}:
			return self._tf.keras.callbacks.ReduceLROnPlateau(**params)

		if type_name in {"tensorboard", "keras_tensorboard"}:
			return self._tf.keras.callbacks.TensorBoard(**params)

		if type_name in {"model_checkpoint", "checkpoint", "keras_checkpoint"}:
			return self._tf.keras.callbacks.ModelCheckpoint(**params)

		if type_name in {"csv_logger", "csvlogger", "keras_csv_logger"}:
			return self._tf.keras.callbacks.CSVLogger(**params)

		if type_name in {"terminate_on_nan", "keras_terminate_on_nan"}:
			return self._tf.keras.callbacks.TerminateOnNaN(**params)

		raise ValueError(
			"Unsupported TensorFlow callback type. Supported: early_stopping, reduce_lr_on_plateau, "
			"tensorboard, model_checkpoint, csv_logger, terminate_on_nan"
		)

	def resolve_callbacks(self, callbacks_cfg: Optional[Sequence[Any]]) -> Optional[List[Any]]:
		if callbacks_cfg is None:
			return None

		if not isinstance(callbacks_cfg, Sequence) or isinstance(callbacks_cfg, (str, bytes)):
			callbacks_cfg = [callbacks_cfg]  # type: ignore[assignment]

		resolved: List[Any] = []
		for raw_cb in callbacks_cfg:
			resolved.append(self.resolve_callback(raw_cb))

		return resolved if resolved else None

	def resolve_optimizer(self, raw_optimizer: Any) -> Any:
		if raw_optimizer is None:
			return self._tf.keras.optimizers.Adam()

		keras_optimizer = self._tf.keras.optimizers.Optimizer
		if isinstance(raw_optimizer, keras_optimizer):
			return raw_optimizer

		if isinstance(raw_optimizer, str):
			return self._tf.keras.optimizers.get(raw_optimizer)

		if isinstance(raw_optimizer, Mapping):
			# Support both Keras-native format:
			#   {"class_name": "Adam", "config": {...}}
			# and compact config format used in this project:
			#   {"type": "adam", "learning_rate": 0.001, ...}
			if "class_name" in raw_optimizer:
				return self._tf.keras.optimizers.get(raw_optimizer)

			type_name, params = self._extract_typed_spec(raw_optimizer, context="optimizer")
			keras_name_map = {
				"adam": "Adam",
				"sgd": "SGD",
				"rmsprop": "RMSprop",
				"adagrad": "Adagrad",
				"adamax": "Adamax",
				"nadam": "Nadam",
				"ftrl": "Ftrl",
			}
			class_name = keras_name_map.get(type_name, type_name)
			return self._tf.keras.optimizers.get(
				{
					"class_name": class_name,
					"config": params,
				}
			)

		if isinstance(raw_optimizer, type) and issubclass(raw_optimizer, keras_optimizer):
			return raw_optimizer()

		if callable(raw_optimizer):
			candidate = raw_optimizer()
			if isinstance(candidate, keras_optimizer):
				return candidate
			raise ValueError(
				"TensorFlow optimizer callable must return tf.keras.optimizers.Optimizer"
			)

		raise ValueError(
			"Unsupported TensorFlow optimizer spec. Use string/object spec, optimizer instance, class, or callable."
		)
