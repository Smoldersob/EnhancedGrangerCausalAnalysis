from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...core.exceptions import ConstraintConfigurationError


class KerasObjectLoader:
	"""Resolve Keras callback and optimizer specs into Keras objects."""

	def __init__(self, keras_m: Any, loading_verbose: bool = False):
		self._keras = keras_m
		self._loading_verbose = bool(loading_verbose)

	@staticmethod
	def resolve_model(model_name: Optional[str], default_model: Any) -> Any:
		"""Resolve a current backend model, accepting either a name or a subclass.

		Supported cases:
		- None / default names -> default_model
		- a subclass of default_model -> return that subclass
		- any other value -> keep the existing default to preserve backward compatibility
		"""
		if model_name is None:
			return default_model

		if isinstance(model_name, type) and issubclass(model_name, default_model):
			return model_name

		normalized = str(model_name).strip().lower()
		if normalized in {"", "default", "base", "standard", "linear", "granger"}:
			return default_model

		return default_model

	def set_loading_verbose(self, value: bool) -> None:
		self._loading_verbose = bool(value)

	def _log(self, label: str, value: Any) -> None:
		if self._loading_verbose:
			print(f"[BackendLoader] KerasObjectLoader: {label} -> {value!r}")

	@staticmethod
	def _extract_typed_spec(raw_spec: Any, *, context: str) -> Tuple[str, Dict[str, Any]]:
		if isinstance(raw_spec, str):
			return raw_spec.strip().lower(), {}

		if not isinstance(raw_spec, Mapping):
			raise ConstraintConfigurationError(f"{context} spec must be a string or object")

		type_name = raw_spec.get("type", raw_spec.get("name", raw_spec.get("kind")))
		if not isinstance(type_name, str) or not type_name.strip():
			raise ConstraintConfigurationError(f"{context} spec requires a non-empty 'type' field")

		params_raw = raw_spec.get("params", {})
		if params_raw is None:
			params_raw = {}
		if not isinstance(params_raw, Mapping):
			raise ConstraintConfigurationError(f"{context} spec 'params' must be an object")

		params = dict(params_raw)
		for key, value in raw_spec.items():
			if key not in {"type", "name", "kind", "params"}:
				params[key] = value

		return type_name.strip().lower(), params

	def resolve_callback(self, raw_callback: Any) -> Any:
		keras_callback_base = self._keras.callbacks.Callback
		if isinstance(raw_callback, keras_callback_base):
			self._log("callback(instance)", raw_callback)
			return raw_callback

		type_name, params = self._extract_typed_spec(raw_callback, context="callback")

		if type_name in {"early_stopping", "keras_early_stopping", "earlystopping"}:
			resolved = self._keras.callbacks.EarlyStopping(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {
			"reduce_lr_on_plateau",
			"reduce_learning_rate",
			"reduce_lr",
			"reducelronplateau",
			"keras_reduce_lr",
		}:
			resolved = self._keras.callbacks.ReduceLROnPlateau(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"tensorboard", "keras_tensorboard"}:
			resolved = self._keras.callbacks.TensorBoard(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"model_checkpoint", "checkpoint", "keras_checkpoint"}:
			resolved = self._keras.callbacks.ModelCheckpoint(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"csv_logger", "csvlogger", "keras_csv_logger"}:
			resolved = self._keras.callbacks.CSVLogger(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"terminate_on_nan", "keras_terminate_on_nan"}:
			resolved = self._keras.callbacks.TerminateOnNaN(**params)
			self._log("callback", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported Keras callback type. Supported: early_stopping, reduce_lr_on_plateau, "
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
			resolved = self._keras.optimizers.Adam()
			self._log("optimizer", resolved)
			return resolved

		keras_optimizer = self._keras.optimizers.Optimizer
		if isinstance(raw_optimizer, keras_optimizer):
			self._log("optimizer(instance)", raw_optimizer)
			return raw_optimizer

		if isinstance(raw_optimizer, str):
			resolved = self._keras.optimizers.get(raw_optimizer)
			self._log("optimizer", resolved)
			return resolved

		if isinstance(raw_optimizer, Mapping):
			# Support Keras-native format:
			#   {"class_name": "Adam", "config": {...}}
			# compact project format:
			#   {"type": "adam", "learning_rate": 0.001}
			# nested params form:
			#   {"type": "adam", "params": {"learning_rate": 0.001}}
			if "class_name" in raw_optimizer:
				resolved = self._keras.optimizers.get(raw_optimizer)
				self._log("optimizer", resolved)
				return resolved

			type_name, params = self._extract_typed_spec(raw_optimizer, context="optimizer")
			# Keep backward compatibility with flat learning_rate declarations.
			if "learning_rate" in raw_optimizer and "learning_rate" not in params and "params" not in raw_optimizer:
				params["learning_rate"] = raw_optimizer["learning_rate"]
			if "params" in raw_optimizer and isinstance(raw_optimizer["params"], Mapping):
				params = dict(raw_optimizer["params"])
				for key, value in raw_optimizer.items():
					if key not in {"type", "name", "kind", "params"}:
						params.setdefault(key, value)

			resolved = self._keras.optimizers.get({"class_name": type_name, "config": params})
			self._log("optimizer", resolved)
			return resolved

		if isinstance(raw_optimizer, type) and issubclass(raw_optimizer, keras_optimizer):
			resolved = raw_optimizer()
			self._log("optimizer", resolved)
			return resolved

		if callable(raw_optimizer):
			candidate = raw_optimizer()
			if isinstance(candidate, keras_optimizer):
				self._log("optimizer(callable)", candidate)
				return candidate
			raise ConstraintConfigurationError(
				"Keras optimizer callable must return tf.keras.optimizers.Optimizer"
			)

		raise ConstraintConfigurationError(
			"Unsupported Keras optimizer spec. Use string/object spec, optimizer instance, class, or callable."
		)