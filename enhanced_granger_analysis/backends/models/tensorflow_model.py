from __future__ import annotations

import importlib
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .base_model import BaseGrangerModel
from ...core.exceptions import (
	BackendNotAvailableError,
	ConstraintConfigurationError,
	RegularizerConfigurationError,
	ModelNotFittedError,
	TrainingError,
)

if find_spec("tensorflow") is not None:
	import tensorflow as tf
else:
	tf = None

@tf.keras.utils.register_keras_serializable(package="EnhancedGrangerCausalAnalysis", name="FeatureMask")
class FeatureMask(tf.keras.layers.Layer):
	"""Untrainable feature mask layer: y = x * mask."""

	def __init__(self, n_features: int, **kwargs: Any) -> None:
		super().__init__(trainable=False, **kwargs)
		self.n_features = n_features

	def build(self, input_shape: Any) -> None:
		self.mask = self.add_weight(
			name="mask",
			shape=(self.n_features,),
			initializer="ones",
			trainable=False,
			dtype=self.dtype,
		)

	def call(self, inputs: Any) -> Any:
		return inputs * self.mask

class TensorFlowGrangerModel(BaseGrangerModel):
	"""TensorFlow implementation of a Granger model with pluggable mask/regularization."""

	def __init__(
		self,
		backend: str = "tensorflow",
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		optimizer: Union[str, Any] = "adam",
		loss: Union[str, Any] = "mse",
		callbacks: Optional[List[Any]] = None,
		epochs: int = 100,
		batch_size: int = 32,
		verbose: int = 0,
		device: Optional[str] = None,
		use_jit: Optional[bool] = None,
		**kwargs: Any,
	) -> None:
		super().__init__(
			backend=backend,
			regularizer=regularizer,
			constraint=constraint,
			callbacks=callbacks or [],
			needs_reinit=False,
		)
		if tf is None:
			raise BackendNotAvailableError(
				"TensorFlow is required to use TensorFlowGrangerModel. "
				"Install tensorflow first."
			)

		self.optimizer = optimizer
		self._optimizer_spec = optimizer
		self._loss_spec = loss

		self.epochs = epochs
		self.batch_size = batch_size
		self.verbose = verbose

		self.device = self._resolve_device(device)
		if use_jit==True and self.device.startswith("/GPU:"):
			self.use_jit = True
		elif use_jit and self.device.startswith("/CPU:"):
			raise ValueError(
				"use_jit=True is unsupported with device='cpu' in this backend. "
				"Use device='gpu' or set use_jit=False."
			)
		else:
			self.use_jit = False

		self.model: Optional[Any] = None
		self._variable_control_layer: Optional[Any] = None
		self._coefficient_layer: Optional[Any] = None
		self._initial_optimizer_state: Optional[tuple[Any, ...]] = None

		self._n_features: Optional[int] = None
		self._n_outputs: Optional[int] = None
		self._X_train: Optional[NDArray[np.float32]] = None
		self._y_train: Optional[NDArray[np.float32]] = None
		self._history: Optional[Any] = None

		self._validate_keras_components()

	def _resolve_device(self, device: Optional[str]) -> str:
		"""Return the target device without modifying the global TensorFlow configuration."""
		gpus = tf.config.list_physical_devices("GPU")

		if device is None or str(device).strip().lower() in {"", "auto"}:
			return "/GPU:0" if gpus else "/CPU:0"

		normalized = str(device).strip().lower()

		if normalized in {"cpu", "cpu-only", "/cpu:0"}:
			return "/CPU:0"

		if normalized in {"gpu", "cuda", "/gpu:0", "/cuda:0"}:
			if not gpus:
				raise BackendNotAvailableError(
					"GPU/CUDA was requested, but TensorFlow does not detect a GPU."
				)
			return "/GPU:0"

		if normalized.startswith(("cuda:", "gpu:", "/cuda:", "/gpu:")):
			raw_index = normalized.rsplit(":", 1)[-1]
			if not raw_index.isdigit():
				raise ValueError(f"Unsupported TensorFlow device spec: {device!r}")

			index = int(raw_index)
			if index >= len(gpus):
				raise BackendNotAvailableError(
					f"Requested GPU {index}, but only {len(gpus)} GPU(s) are available."
				)
			return f"/GPU:{index}"

		raise ValueError(
			"device must be one of: None, 'auto', 'cpu', 'gpu', 'cuda', "
			"'cuda:N' or 'gpu:N'."
		)

	def _validate_keras_components(self) -> None:
		"""Validate optional regularizer/constraint against Keras base classes."""
		keras_regularizer = tf.keras.regularizers.Regularizer
		keras_constraint = tf.keras.constraints.Constraint

		if self.regularizer is not None and not isinstance(self.regularizer, keras_regularizer):
			raise RegularizerConfigurationError(
				"regularizer must inherit from tf.keras.regularizers.Regularizer "
				"for TensorFlowGrangerModel"
			)

		if self.constraint is not None and not isinstance(self.constraint, keras_constraint):
			raise ConstraintConfigurationError(
				"constraint must inherit from tf.keras.constraints.Constraint "
				"for TensorFlowGrangerModel"
			)

		keras_callback = tf.keras.callbacks.Callback
		if not isinstance(self.callbacks, list):
			raise ConstraintConfigurationError(
				"callbacks must be a list of tf.keras.callbacks.Callback objects"
			)
		for callback in self.callbacks:
			if not isinstance(callback, keras_callback):
				raise ConstraintConfigurationError(
					"All callbacks must inherit from tf.keras.callbacks.Callback"
				)

	def _normalize_optimizer_spec(self, spec: Any) -> Any:
		"""Convert generic optimizer config into a Keras-deserializable mapping."""
		if not isinstance(spec, dict):
			return spec

		if "class_name" in spec and isinstance(spec.get("config"), dict):
			return spec

		optimizer_name = spec.get("type") or spec.get("class_name") or spec.get("name") or "adam"
		config: Dict[str, Any] = {}

		params = spec.get("params")
		if isinstance(params, dict):
			config.update(params)

		native_config = spec.get("config")
		if isinstance(native_config, dict):
			config.update(native_config)

		for key, value in spec.items():
			if key not in {"type", "class_name", "name", "params", "config"}:
				config[key] = value

		if "learning_rate" in spec and "learning_rate" not in config:
			config["learning_rate"] = spec["learning_rate"]

		name = str(optimizer_name).strip()
		alias_map = {
			"adam": "Adam",
			"adadelta": "Adadelta",
			"adagrad": "Adagrad",
			"adamax": "Adamax",
			"ftrl": "Ftrl",
			"nadam": "Nadam",
			"rmsprop": "RMSprop",
			"sgd": "SGD",
		}
		canonical_name = alias_map.get(name.lower(), name)

		return {"class_name": canonical_name, "config": config}

	def _build_optimizer(self) -> Any:
		"""Create a fresh Keras optimizer instance from optimizer spec."""
		keras_optimizer = tf.keras.optimizers.Optimizer

		spec = self._optimizer_spec
		if isinstance(spec, str):
			return tf.keras.optimizers.get(spec)

		if isinstance(spec, dict):
			return tf.keras.optimizers.deserialize(self._normalize_optimizer_spec(spec))
		if isinstance(spec, type) and issubclass(spec, keras_optimizer):
			return spec()

		if isinstance(spec, keras_optimizer):
			return spec.__class__.from_config(spec.get_config())

		if callable(spec):
			candidate = spec()
			if isinstance(candidate, keras_optimizer):
				return candidate
			raise ConstraintConfigurationError(
				"optimizer callable must return tf.keras.optimizers.Optimizer"
			)

		raise ConstraintConfigurationError(
			"optimizer must be string, keras optimizer, keras optimizer class, dict, or callable"
		)

	def _build_loss(self) -> Any:
		"""Resolve loss spec to Keras-compatible loss object/callable."""
		return tf.keras.losses.get(self._loss_spec)

	def _capture_initial_optimizer_state(self) -> None:
		"""Build optimizer slots and save pristine state on the active device."""
		if self.model is None or self.model.optimizer is None:
			return

		optimizer = self.model.optimizer
		optimizer.build(self.model.trainable_variables)

		variables = optimizer.variables
		if callable(variables):
			variables = variables()

		with tf.device(self.device):
			self._initial_optimizer_state = [
				tf.identity(variable.value)
				for variable in variables
			]

	def _reset_optimizer_state(self) -> None:
		if self.model is None or self.model.optimizer is None:
			return

		if self._initial_optimizer_state is None:
			self._capture_initial_optimizer_state()

		variables = self.model.optimizer.variables
		if callable(variables):
			variables = variables()

		if len(variables) != len(self._initial_optimizer_state):
			raise TrainingError("Optimizer variable layout changed.")

		for variable, initial_value in zip(variables, self._initial_optimizer_state):
			variable.assign(initial_value)

	def initialize(
		self,
		data: NDArray[np.float32],
		lags: Optional[int] = None,
		**kwargs: Any,
	) -> None:
		"""Initialize model using lagged features prepared externally (e.g. LagEngine)."""
		self._validate_keras_components()

		X = np.asarray(data, dtype=np.float32)
		y_raw = kwargs.get("targets")
		if y_raw is None:
			raise TrainingError(
				"initialize requires precomputed targets via targets=<ndarray>. "
				"Lagged features should be prepared by LagEngine."
			)

		y = np.asarray(y_raw, dtype=np.float32)
		if X.ndim != 2:
			raise TrainingError("Expected 2D lagged feature matrix with shape (n_samples, n_lagged_features)")
		if y.ndim == 1:
			y = y[:, np.newaxis]
		if y.ndim != 2:
			raise TrainingError("targets must be 1D or 2D array")
		if X.shape[0] != y.shape[0]:
			raise TrainingError("Lagged features and targets must have the same number of rows")

		n_features = X.shape[1]
		n_outputs = y.shape[1]

		with tf.device(self.device):
			inputs = tf.keras.Input(shape=(n_features,), dtype=tf.float32)

			variable_control_layer = FeatureMask(
				n_features=n_features,
				name="variable_control",
				dtype=tf.float32,
			)
			selected_features = variable_control_layer(inputs)

			coefficient_layer = tf.keras.layers.Dense(
				n_outputs,
				use_bias=True,
				kernel_constraint=self.constraint,
				kernel_regularizer=self.regularizer,
				name="coefficients",
				dtype=tf.float32,
			)
			outputs = coefficient_layer(selected_features)

			self.model = tf.keras.Model(inputs, outputs=outputs, name="granger_model")

			self.model.compile(
				optimizer = self._build_optimizer(),
				loss = self._build_loss(),
				jit_compile=self.use_jit,
			)
		
			self._capture_initial_optimizer_state()

		self._variable_control_layer = variable_control_layer
		self._coefficient_layer = coefficient_layer
		self._n_features = n_features
		self._n_outputs = n_outputs
		self._X_train = X
		self._y_train = y
		self._fitted = False

	def fit(self, return_history: bool = False) -> Dict[str, Any]|None:
		"""Fit model and return a minimal result dictionary aligned with BaseGrangerModel."""
		if self.model is None or self._X_train is None or self._y_train is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		# Reset optimizer state between fits without costly re-compile.
		self._reset_optimizer_state()
		batch_size = self.batch_size or self._X_train.shape[0]

		try:
			self._history = self.model.fit(
				self._X_train,
				self._y_train,
				epochs=self.epochs,
				batch_size=batch_size,
				callbacks=self.callbacks,
				verbose=self.verbose,
				shuffle=False,
			)
		except Exception as exc:  # pragma: no cover - backend runtime errors
			if self._is_gpu_dnn_init_error(exc):
				raise TrainingError(
					"TensorFlow GPU runtime failed during DNN initialization. "
					"Run in stable CPU mode by setting CGA_TF_FORCE_CPU=1 "
					"or (on WSL) leave CGA_TF_USE_GPU unset. "
					f"Original error: {exc}"
				) from exc
			else:
				raise TrainingError(f"TensorFlow training failed: {exc}") from exc

		self._fitted = True

		if return_history:
			forecasts = self.model(self._X_train, training=False).numpy().astype(np.float32)

			final_loss = (
				float(self._history.history["loss"][-1])
				if self._history is not None and "loss" in self._history.history
				else float("nan")
			)

			loss_history = self._history.history.get("loss", []) if self._history is not None else []
			epoch_indices = list(getattr(self._history, "epoch", [])) if self._history is not None else []
			epochs_ran = len(epoch_indices) if epoch_indices else len(loss_history)
			stop_reason = "callback_stop" if epochs_ran < self.epochs else "max_epochs_reached"

			return {
				"test_statistic": final_loss,
				"p_value": np.nan,
				"weights": self.get_weights(),
				"forecasts": forecasts,
				"history": {
					"loss": loss_history,
					"stop_reason": stop_reason,
				},
			}
		else:
			return 

	@staticmethod
	def _is_gpu_dnn_init_error(exc: Exception) -> bool:
		msg = str(exc).lower()
		return (
			"dnn library initialization failed" in msg
			or "cudnn_status_not_initialized" in msg
			or "failedpreconditionerror" in msg and "cuda" in msg
		)

	def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
		"""Generate predictions from fitted TensorFlow model."""
		if not self._fitted or self.model is None or self._n_features is None:
			raise ModelNotFittedError("Model is not fitted. Call fit(...) first.")

		if X.ndim != 2:
			raise TrainingError("X must be a 2D array")
		if X.shape[1] != self._n_features:
			raise TrainingError(
				f"X has {X.shape[1]} features, expected {self._n_features}"
			)

		with tf.device(self.device):
			pred = self.model(
				tf.convert_to_tensor(X, dtype=tf.float32),
				training=False,
			)

		return pred.numpy().astype(np.float32, copy=False)

	def set_weights(
		self, weights: Union[NDArray[np.float32], List[NDArray[np.float32]]]
	) -> None:
		"""Set coefficient-layer kernel (and optional bias) weights."""
		if self._coefficient_layer is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		if isinstance(weights, list):
			if len(weights) == 1:
				with tf.device(self.device):
					current_weights = self._coefficient_layer.get_weights()
					self._coefficient_layer.set_weights([weights[0], current_weights[1]])
			elif len(weights) == 2:
				with tf.device(self.device):
					self._coefficient_layer.set_weights([weights[0], weights[1]])
			else:
				raise TrainingError("weights list must contain kernel or [kernel, bias]")
			return

	def get_weights(self) -> List[NDArray[np.float32]]:
		"""Return coefficient-layer weights as a single matrix in a one-element list."""
		if self._coefficient_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		coeff_weights = self._coefficient_layer.get_weights()
		if not coeff_weights:
			return []
		else:
			return coeff_weights
		
	def omit_variables(self, variable_indices: List[int]) -> None:
		"""Set selected variables to zero in the non-trainable diagonal control layer."""
		if self._variable_control_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		with tf.device(self.device):
			variable_mask = np.ones(self._n_features, dtype=np.float32)

			indices = np.asarray(variable_indices, dtype=np.intp)
			if np.any(indices < 0) or np.any(indices >= self._n_features):
				raise TrainingError(
					f"Variable indices must belong to [0, {self._n_features - 1}]"
				)

			variable_mask[indices] = 0.0
			self._variable_control_layer.mask.assign(variable_mask)

	def set_regularizer(self, regularizer: Any) -> None:
		"""Set regularizer with Keras type validation."""
		self.regularizer = regularizer
		self._validate_keras_components()

	def set_constraint(self, constraint: Any) -> None:
		"""Set constraint with Keras type validation."""
		self.constraint = constraint
		self._validate_keras_components()


	def reset_callbacks(
		self,
		*,
		run_name: Optional[str] = None,
		tensorboard_root_dir: Optional[Union[str, os.PathLike[str]]] = None,
		callback_updates: Optional[
			Mapping[Union[int, str, type], Mapping[str, Any]]
		] = None,
	) -> None:
		"""
		Prepare callbacks for a fully independent subsequent fit() call.

		This method does not rebuild the model, recompile it, or copy
		callback instances. It assumes that new initial model weights are
		set separately via set_weights(...).

		Parameters
		----------
		run_name:
			Name of the new TensorBoard run. If a TensorBoard callback is
			configured, its log_dir is changed to:
			<tensorboard_root_dir>/<run_name>.
		tensorboard_root_dir:
			Parent directory for TensorBoard runs. If omitted, the parent
			directory of the current TensorBoard log_dir is used.
		callback_updates:
			Configuration updates for matching callbacks. A selector can be:
			- callback index in self.callbacks,
			- callback class name, e.g. "EarlyStopping",
			- callback type, e.g. tf.keras.callbacks.EarlyStopping.

			Example:
			{
				tf.keras.callbacks.EarlyStopping: {
					"patience": 20,
					"min_delta": 1e-5,
				},
				"ReduceLROnPlateau": {
					"factor": 0.5,
					"patience": 4,
				},
			}
		reset_optimizer_learning_rate:
			Restore the original optimizer learning rate. This is important
			after a ReduceLROnPlateau callback changed it during training.
		"""
		if self.model is None:
			raise ModelNotFittedError(
				"Model is not initialized. Call initialize(...) first."
			)

		self._validate_keras_components()

		has_callback_updates = bool(callback_updates)

		for index, callback in enumerate(self.callbacks):
			if has_callback_updates:
				updates = self._get_callback_updates(
					callback=callback,
					index=index,
					callback_updates=callback_updates,
				)
				if updates:
					self._apply_callback_updates(callback, updates)

			if isinstance(callback, tf.keras.callbacks.TensorBoard):
				self._reset_tensorboard_callback(
					callback=callback,
					run_name=run_name,
					root_dir=tensorboard_root_dir,
				)
			else:
				self._reset_callback_runtime_state(callback)

		# Keras should resets this, but it's better to makes sure
		self.model.stop_training = False

	def _get_callback_updates(
		self,
		*,
		callback: Any,
		index: int,
		callback_updates: Optional[
			Mapping[Union[int, str, type], Mapping[str, Any]]
		],
	) -> Dict[str, Any]:
		"""Merge all configuration updates that match a callback."""
		if not callback_updates:
			return {}

		updates: Dict[str, Any] = {}

		for selector, values in callback_updates.items():
			if not isinstance(values, Mapping):
				raise ConstraintConfigurationError(
					"Each callback_updates value must be a mapping of "
					"attribute names to values."
				)

			matches = (
				selector == index
				or selector == callback.__class__.__name__
				or (
					isinstance(selector, type)
					and isinstance(callback, selector)
				)
			)

			if matches:
				updates.update(values)

		return updates

	def _apply_callback_updates(
		self,
		callback: Any,
		updates: Mapping[str, Any],
	) -> None:
		"""
		Update only existing public callback attributes.

		Private attributes are intentionally excluded because they are
		version-dependent runtime internals rather than stable configuration.
		"""
		for attribute, value in updates.items():
			if attribute.startswith("_"):
				raise ConstraintConfigurationError(
					f"Cannot update private callback attribute: {attribute!r}"
				)

			if not hasattr(callback, attribute):
				raise ConstraintConfigurationError(
					f"{callback.__class__.__name__} has no configurable "
					f"attribute {attribute!r}"
				)

			setattr(callback, attribute, value)

	def _reset_callback_runtime_state(self, callback: Any) -> None:
		"""
		Clear known callback runtime state without reconstructing the callback
		or copying its configuration.
		"""
		reset_state = getattr(callback, "reset_state", None)
		if callable(reset_state):
			reset_state()
			return

		if isinstance(callback, tf.keras.callbacks.EarlyStopping):
			callback.wait = 0
			callback.stopped_epoch = 0
			callback.best_epoch = 0
			callback.best_weights = None

			# on_train_begin() initializes best and monitor_op according to
			# the active TensorFlow/Keras version and callback configuration.
			callback.on_train_begin(logs=None)
			return

		if isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau):
			callback.wait = 0
			callback.cooldown_counter = 0
			callback.best = np.inf
			callback.on_train_begin(logs=None)
			return

		if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
			callback.best = np.inf
			callback.on_train_begin(logs=None)
			return

		# Custom callbacks without reset_state() intentionally retain their
		# state. Resetting __dict__ could remove model references, file
		# handles, configuration values, or other persistent resources.

	def _reset_tensorboard_callback(
		self,
		*,
		callback: Any,
		run_name: Optional[str],
		root_dir: Optional[Union[str, os.PathLike[str]]],
	) -> None:
		"""
		Assign a new TensorBoard run directory and discard cached writer state.

		TensorBoard will create fresh writers during the next model.fit() call.
		"""
		if run_name is not None:
			base_dir = (
				Path(root_dir)
				if root_dir is not None
				else Path(callback.log_dir).parent
			)
			callback.log_dir = str(base_dir / run_name)

		# TensorBoard internals vary between TensorFlow/Keras releases.
		# Clear only known writer and counter fields when they are present.
		for attribute in (
			"_train_writer",
			"_val_writer",
			"_writers",
			"_train_step",
			"_val_step",
			"_global_train_batch",
			"_global_test_batch",
		):
			if hasattr(callback, attribute):
				setattr(callback, attribute, None)