from __future__ import annotations

import importlib
import os
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Union

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
		if use_jit is None:
			self.use_jit = self.device.startswith("/GPU:")
		else:
			self.use_jit = bool(use_jit)

		if self.device.startswith("/CPU:") and self.use_jit:
			raise ValueError(
				"use_jit=True is unsupported with device='cpu' in this backend. "
				"Use device='gpu' or set use_jit=False."
			)

		self.model: Optional[Any] = None
		self._variable_control_layer: Optional[Any] = None
		self._coefficient_layer: Optional[Any] = None

		self._n_features: Optional[int] = None
		self._n_outputs: Optional[int] = None
		self._variable_mask: Optional[NDArray[np.float32]] = None
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

	def _build_optimizer(self) -> Any:
		"""Create a fresh Keras optimizer instance from optimizer spec."""
		keras_optimizer = tf.keras.optimizers.Optimizer

		spec = self._optimizer_spec
		if isinstance(spec, str) or isinstance(spec, dict):
			return tf.keras.optimizers.get(spec)

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
		"""Build optimizer slots and save a fresh optimizer state without a training step."""
		if self.model is None or self.model.optimizer is None:
			return

		optimizer = self.model.optimizer
		optimizer.build(self.model.trainable_variables)

		variables = optimizer.variables
		if callable(variables):
			variables = variables()

		self._initial_optimizer_weights = [
			np.array(variable.numpy(), copy=True)
			for variable in variables
		]


	def _restore_initial_optimizer_state(self) -> None:
		"""Restores the optimizer state saved after its initialization."""
		with tf.device(self.device):
			if self.model is None or self.model.optimizer is None:
				return

			if self._initial_optimizer_weights is None:
				self._capture_initial_optimizer_state()

			optimizer = self.model.optimizer
			variables = optimizer.variables
			if callable(variables):
				variables = variables()

			if len(variables) != len(self._initial_optimizer_weights):
				raise TrainingError(
					"Optimizer variable layout changed; cannot restore initial optimizer state."
				)

			for variable, initial_value in zip(variables, self._initial_optimizer_weights):
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
		variable_mask = np.ones(n_features, dtype=np.float32)
		identity_kernel = np.eye(n_features, dtype=np.float32)

		with tf.device(self.device):
			variable_control_layer = FeatureMask(
				n_features=n_features,
				name="variable_control",
				dtype=tf.float32,
			)

			coefficient_layer = tf.keras.layers.Dense(
				units=n_outputs,
				use_bias=True,
				kernel_constraint=self.constraint,
				kernel_regularizer=self.regularizer,
				name="coefficients",
				dtype=tf.float32,
			)

			self.model = tf.keras.Sequential(
				[
					tf.keras.layers.Input(shape=(n_features,), dtype=tf.float32),
					variable_control_layer,
					coefficient_layer,
				],
				name="tensorflow_granger_model",
			)

			optimizer = self._build_optimizer()
			loss = self._build_loss()

			self.model.compile(
				optimizer=optimizer,
				loss=loss,
				jit_compile=self.use_jit,
			)
	
			self._capture_initial_optimizer_state()	

		self._variable_control_layer = variable_control_layer
		self._coefficient_layer = coefficient_layer
		self._n_features = n_features
		self._n_outputs = n_outputs
		self._variable_mask = variable_mask
		self._X_train = X
		self._y_train = y
		self._fitted = False

	def fit(self) -> Dict[str, Any]:
		"""Fit model and return a minimal result dictionary aligned with BaseGrangerModel."""
		if self.model is None or self._X_train is None or self._y_train is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		# Reset optimizer state between fits without costly re-compile.
		self._restore_initial_optimizer_state()

		try:
			if self.batch_size is None:
				self.batch_size = self._X_train.shape[0]
				
			self._history = self.model.fit(
				self._X_train,
				self._y_train,
				epochs=self.epochs,
				batch_size=self.batch_size,
				callbacks=self.callbacks,
				verbose=self.verbose,
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
		forecasts = np.asarray(self.model.predict(self._X_train, verbose=0), dtype=np.float32)

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

		X_arr = np.asarray(X, dtype=np.float32)
		if X_arr.ndim != 2:
			raise TrainingError("X must be a 2D array")
		if X_arr.shape[1] != self._n_features:
			raise TrainingError(
				f"X has {X_arr.shape[1]} features, expected {self._n_features}"
			)

		with tf.device(self.device):
			pred = self.model.predict(X_arr, verbose=0)
		return np.asarray(pred, dtype=np.float64)

	def set_weights(
		self, weights: Union[NDArray[np.float32], List[NDArray[np.float32]]]
	) -> None:
		"""Set coefficient-layer kernel (and optional bias) weights."""
		if self._coefficient_layer is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		current_weights = self._coefficient_layer.get_weights()
		if not current_weights:
			raise TrainingError("Coefficient layer is not built yet.")

		if isinstance(weights, list):
			if len(weights) == 1:
				with tf.device(self.device):
					self._coefficient_layer.set_weights([weights[0], current_weights[1]])
			elif len(weights) == 2:
				with tf.device(self.device):
					self._coefficient_layer.set_weights([weights[0], weights[1]])
			else:
				raise TrainingError("weights list must contain kernel or [kernel, bias]")
			return

		with tf.device(self.device):
			self._coefficient_layer.set_weights([weights, current_weights[1]])

	def get_weights(self) -> List[NDArray[np.float32]]:
		"""Return coefficient-layer weights as a single matrix in a one-element list."""
		if self._coefficient_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		coeff_weights = self._coefficient_layer.get_weights()
		if not coeff_weights:
			return []

		kernel = coeff_weights[0]
		bias = coeff_weights[1] if len(coeff_weights) > 1 else np.zeros(self._n_outputs, dtype=np.float32)
		return [np.asarray(kernel, dtype=np.float32), np.asarray(bias, dtype=np.float32)]

	def omit_variables(self, variable_indices: List[int]) -> None:
		"""Set selected variables to zero in the non-trainable diagonal control layer."""
		if self._variable_control_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		with tf.device(self.device):
			self._variable_mask = np.ones(self._n_features, dtype=np.float64)

			indices = np.asarray(variable_indices, dtype=np.intp)
			if np.any(indices < 0) or np.any(indices >= self._n_features):
				raise TrainingError(
					f"Variable indices must belong to [0, {self._n_features - 1}]"
				)

			self._variable_mask[indices] = 0.0
			self._variable_control_layer.mask.assign(self._variable_mask)

	def set_regularizer(self, regularizer: Any) -> None:
		"""Set regularizer with Keras type validation."""
		self.regularizer = regularizer
		self._validate_keras_components()

	def set_constraint(self, constraint: Any) -> None:
		"""Set constraint with Keras type validation."""
		self.constraint = constraint
		self._validate_keras_components()

