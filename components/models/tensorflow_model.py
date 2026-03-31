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
	ModelNotFittedError,
	TrainingError,
)

if find_spec("tensorflow") is not None:
	tf = importlib.import_module("tensorflow")

	_force_cpu_env = os.getenv("CGA_TF_FORCE_CPU", "").strip().lower()
	_use_gpu_env = os.getenv("CGA_TF_USE_GPU", "").strip().lower()
	_is_wsl = bool(os.getenv("WSL_DISTRO_NAME"))
	_force_cpu = _force_cpu_env in {"1", "true", "yes", "on"}
	_explicit_use_gpu = _use_gpu_env in {"1", "true", "yes", "on"}
	_prefer_cpu = _force_cpu or (_is_wsl and not _explicit_use_gpu)

	if _prefer_cpu:
		try:
			tf.config.set_visible_devices([], "GPU")
		except Exception:
			pass
else:  # pragma: no cover - runtime dependency check
	tf = None


class TensorFlowGrangerModel(BaseGrangerModel):
	"""TensorFlow implementation of a Granger model with pluggable mask/regularization."""

	def __init__(
		self,
		backend: str = "tensorflow",
		scaler: Optional[Any] = None,
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		optimizer: Union[str, Any] = "adam",
		loss: Union[str, Any] = "mse",
		callbacks: Optional[List[Any]] = None,
		epochs: int = 100,
		batch_size: int = 32,
		verbose: int = 0,
	) -> None:
		super().__init__(
			backend=backend,
			scaler=scaler,
			regularizer=regularizer,
			constraint=constraint,
		)
		if tf is None:
			raise BackendNotAvailableError(
				"TensorFlow is required to use TensorFlowGrangerModel. "
				"Install tensorflow first."
			)

		self.optimizer = optimizer
		self.loss = loss
		self._optimizer_spec = optimizer
		self._loss_spec = loss
		self.callbacks = callbacks or []

		self.epochs = epochs
		self.batch_size = batch_size
		self.verbose = verbose

		self.model: Optional[Any] = None
		self._variable_control_layer: Optional[Any] = None
		self._coefficient_layer: Optional[Any] = None

		self._n_features: Optional[int] = None
		self._n_outputs: Optional[int] = None
		self._variable_mask: Optional[NDArray[np.float64]] = None
		self._X_train: Optional[NDArray[np.float64]] = None
		self._y_train: Optional[NDArray[np.float64]] = None
		self._history: Optional[Any] = None

		self._validate_keras_components()

	def _validate_keras_components(self) -> None:
		"""Validate optional regularizer/constraint against Keras base classes."""
		keras_regularizer = tf.keras.regularizers.Regularizer
		keras_constraint = tf.keras.constraints.Constraint

		if self.regularizer is not None and not isinstance(self.regularizer, keras_regularizer):
			raise ConstraintConfigurationError(
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

	def _reset_optimizer_state(self) -> None:
		"""Reset optimizer internal state without recompiling the model."""
		if self.model is None or getattr(self.model, "optimizer", None) is None:
			return

		optimizer = self.model.optimizer
		# Keras optimizers expose state variables (iteration + slots).
		# Zeroing them is much cheaper than re-compiling the full model graph.
		for var in optimizer.variables:
			var.assign(tf.zeros_like(var))

	def initialize(
		self,
		data: NDArray[np.float64],
		lags: Optional[int] = None,
		**kwargs: Any,
	) -> None:
		"""Initialize model using lagged features prepared externally (e.g. LagEngine)."""
		self._validate_keras_components()

		X = np.asarray(data, dtype=np.float64)
		y_raw = kwargs.get("targets")
		if y_raw is None:
			raise TrainingError(
				"initialize requires precomputed targets via targets=<ndarray>. "
				"Lagged features should be prepared by LagEngine."
			)

		y = np.asarray(y_raw, dtype=np.float64)
		if X.ndim != 2:
			raise TrainingError("Expected 2D lagged feature matrix with shape (n_samples, n_lagged_features)")
		if y.ndim == 1:
			y = y[:, np.newaxis]
		if y.ndim != 2:
			raise TrainingError("targets must be 1D or 2D array")
		if X.shape[0] != y.shape[0]:
			raise TrainingError("Lagged features and targets must have the same number of rows")

		if self.scaler is not None:
			X = self.scaler.fit_transform(X)

		n_features = X.shape[1]
		n_outputs = y.shape[1]
		variable_mask = np.ones(n_features, dtype=np.float64)
		identity_kernel = np.eye(n_features, dtype=np.float64)

		variable_control_layer = tf.keras.layers.Dense(
			units=n_features,
			use_bias=False,
			trainable=False,
			kernel_initializer=tf.keras.initializers.Constant(identity_kernel),
			name="variable_control",
			dtype=tf.float64,
		)

		coefficient_layer = tf.keras.layers.Dense(
			units=n_outputs,
			use_bias=True,
			kernel_constraint=self.constraint,
			kernel_regularizer=self.regularizer,
			name="coefficients",
			dtype=tf.float64,
		)

		self.model = tf.keras.Sequential(
			[
				tf.keras.layers.Input(shape=(n_features,), dtype=tf.float64),
				variable_control_layer,
				coefficient_layer,
			],
			name="tensorflow_granger_model",
		)
		self.model.compile(optimizer=self._build_optimizer(), loss=self._build_loss())

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
		self._reset_optimizer_state()

		try:
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
		forecasts = self.model.predict(self._X_train, verbose=0)

		final_loss = (
			float(self._history.history["loss"][-1])
			if self._history is not None and "loss" in self._history.history
			else float("nan")
		)

		return {
			"test_statistic": final_loss,
			"p_value": np.nan,
			"weights": self.get_weights(),
			"forecasts": forecasts,
			"history": self._history.history if self._history is not None else {},
		}

	@staticmethod
	def _is_gpu_dnn_init_error(exc: Exception) -> bool:
		msg = str(exc).lower()
		return (
			"dnn library initialization failed" in msg
			or "cudnn_status_not_initialized" in msg
			or "failedpreconditionerror" in msg and "cuda" in msg
		)

	def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Generate predictions from fitted TensorFlow model."""
		if not self._fitted or self.model is None or self._n_features is None:
			raise ModelNotFittedError("Model is not fitted. Call fit(...) first.")

		X_arr = np.asarray(X, dtype=np.float64)
		if X_arr.ndim != 2:
			raise TrainingError("X must be a 2D array")
		if X_arr.shape[1] != self._n_features:
			raise TrainingError(
				f"X has {X_arr.shape[1]} features, expected {self._n_features}"
			)

		if self._variable_mask is not None:
			X_arr = X_arr * self._variable_mask[np.newaxis, :]

		pred = self.model.predict(X_arr, verbose=0)
		return np.asarray(pred, dtype=np.float64)

	def set_weights(
		self, weights: Union[NDArray[np.float64], List[NDArray[np.float64]]]
	) -> None:
		"""Set coefficient-layer kernel (and optional bias) weights."""
		if self._coefficient_layer is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		current_weights = self._coefficient_layer.get_weights()
		if not current_weights:
			raise TrainingError("Coefficient layer is not built yet.")

		if isinstance(weights, list):
			if len(weights) == 1:
				self._coefficient_layer.set_weights([weights[0], current_weights[1]])
			elif len(weights) == 2:
				self._coefficient_layer.set_weights([weights[0], weights[1]])
			else:
				raise TrainingError("weights list must contain kernel or [kernel, bias]")
			return

		self._coefficient_layer.set_weights([weights, current_weights[1]])

	def get_weights(self) -> List[NDArray[np.float64]]:
		"""Return coefficient-layer weights as a single matrix in a one-element list."""
		if self._coefficient_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		coeff_weights = self._coefficient_layer.get_weights()
		if not coeff_weights:
			return []

		kernel = coeff_weights[0]
		return [np.asarray(kernel, dtype=np.float64)]

	def omit_variables(self, variable_indices: List[int]) -> None:
		"""Set selected variables to zero in the non-trainable diagonal control layer."""
		if self._variable_control_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		if self._variable_mask is None:
			self._variable_mask = np.ones(self._n_features, dtype=np.float64)

		for idx in variable_indices:
			if idx < 0 or idx >= self._n_features:
				raise TrainingError(
					f"Variable index {idx} out of range [0, {self._n_features - 1}]"
				)
			self._variable_mask[idx] = 0.0

		diagonal_kernel = np.diag(self._variable_mask).astype(np.float64)
		self._variable_control_layer.set_weights([diagonal_kernel])

	def set_regularizer(self, regularizer: Any) -> None:
		"""Set regularizer with Keras type validation."""
		self.regularizer = regularizer
		self._validate_keras_components()

	def set_constraint(self, constraint: Any) -> None:
		"""Set constraint with Keras type validation."""
		self.constraint = constraint
		self._validate_keras_components()

	def hyperoptimize(
		self,
		reg_param_grid: Dict[str, List[Any]],
		n_trials: int = 50,
	) -> Dict[str, Any]:
		"""Return explicit no-op hyperoptimization result for this model."""
		return {
			"best_params": {},
			"best_score": np.nan,
			"trial_results": [],
			"n_trials_requested": n_trials,
			"reg_param_grid": reg_param_grid,
			"message": "TensorFlowGrangerModel nie posiada parametrów do hiperoptymalizacji.",
		}

