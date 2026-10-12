from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Optional, Type, Mapping
import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel, _preprocess_data

from .base_model import BaseGrangerModel
from ..callbacks.base_callback import Callback
from ..callbacks.reset_utils import (
    clone_custom_callback,
    reset_callback_list,
)
from ...core.exceptions import (
	ConstraintConfigurationError,
	RegularizerConfigurationError,
	ModelNotFittedError,
	TrainingError,
)


class _OptimizerProxy:
	"""Small optimizer-like adapter exposing param_groups for callback compatibility."""

	def __init__(self, lr: float):
		self.param_groups: List[Dict[str, float]] = [{"lr": float(lr)}]


class _ArrayAdapter:
	"""Tensor-like adapter used by callbacks that expect detach().cpu()."""

	def __init__(self, array: NDArray[np.float64]):
		self._array = array

	def detach(self) -> "_ArrayAdapter":
		return self

	def cpu(self) -> NDArray[np.float64]:
		return self._array


class _CoefficientLayerAdapter:
	"""Layer-like adapter exposing a weight attribute used by logging callbacks."""

	def __init__(self, model: "ScikitConstrainedGrangerModel"):
		self._model = model

	@property
	def weight(self) -> _ArrayAdapter:
		coef = np.asarray(self._model.coef_, dtype=np.float64)
		return _ArrayAdapter(coef)


class _VariableControlLayer:
	"""Non-trainable diagonal masking layer for scikit-based model."""

	def __init__(self, n_features: int):
		self.trainable = False
		self._n_features = n_features
		self._mask = np.ones(n_features, dtype=np.float64)
		self.weight = np.diag(self._mask).astype(np.float64)

	def reset(self) -> None:
		self._mask = np.ones(self._n_features, dtype=np.float64)
		self.weight = np.diag(self._mask).astype(np.float64)

	def omit(self, variable_indices: List[int]) -> None:
		for idx in variable_indices:
			if idx < 0 or idx >= self._n_features:
				raise TrainingError(f"Variable index {idx} out of range [0, {self._n_features - 1}]")
			self._mask[idx] = 0.0
		self.weight = np.diag(self._mask).astype(np.float64)

	def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
		return x @ self.weight

	@property
	def mask(self) -> NDArray[np.float64]:
		return self._mask.copy()


class ScikitConstrainedGrangerModel(LinearModel, RegressorMixin, BaseGrangerModel):
	"""Scikit-compatible Granger model with gradient descent, callbacks, constraints and regularization."""

	_NEWTON_EPS = 1e-12

	def __init__(
		self,
		backend: str = "sklearn",
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		callbacks: List[Callback] = None,
		fit_intercept: bool = True,
		learning_rate: float = 1.0,
		max_iter: int = 1000,
		tol: float = 1e-8,
		batch_size: Optional[int] = None,
		verbose: int = 0,
		**kwargs: Any
	):
		BaseGrangerModel.__init__(
			self,
			backend=backend,
			regularizer=regularizer,
			constraint=constraint,
			callbacks=callbacks or [],
			needs_reinit=False,
		)

		if learning_rate <= 0:
			raise ConstraintConfigurationError("learning_rate must be > 0")
		if max_iter <= 0:
			raise ConstraintConfigurationError("max_iter must be > 0")
		if tol < 0:
			raise ConstraintConfigurationError("tol must be >= 0")
		if batch_size is not None and batch_size <= 0:
			raise ConstraintConfigurationError("batch_size must be > 0 when provided")

		self.fit_intercept = fit_intercept
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.tol = tol
		self.batch_size = batch_size
		self.verbose = verbose

		self._n_features: Optional[int] = None
		self._n_outputs: Optional[int] = None
		self._variable_mask: Optional[NDArray[np.float64]] = None
		self._X_train: Optional[NDArray[np.float64]] = None
		self._y_train: Optional[NDArray[np.float64]] = None
		self._loss_history: List[float] = []
		self._optimizer_proxy = _OptimizerProxy(self.learning_rate)
		self._variable_control_layer: Optional[_VariableControlLayer] = None
		self._coefficient_layer = _CoefficientLayerAdapter(self)

		self._validate_components()

	def _validate_components(self) -> None:
		if self.regularizer is not None and not callable(self.regularizer):
			raise RegularizerConfigurationError("regularizer must be callable for ScikitConstrainedGrangerModel")

		if self.constraint is not None and not callable(self.constraint):
			raise ConstraintConfigurationError("constraint must be callable for ScikitConstrainedGrangerModel")

		if not isinstance(self.callbacks, list):
			raise ConstraintConfigurationError("callbacks must be a list of Callback objects")
		for callback in self.callbacks:
			if not isinstance(callback, Callback):
				raise ConstraintConfigurationError(
					"All callbacks must inherit from callbacks.base_callback.Callback"
				)

	def _run_callback_hook(self, hook_name: str, state: Dict[str, Any]) -> bool:
		for callback in self.callbacks:
			hook = getattr(callback, hook_name)
			response = hook(state)
			if response is False:
				state["stop_training"] = True
				if not state.get("stop_reason"):
					state["stop_reason"] = f"{callback.__class__.__name__}:{hook_name}"
				return False
		return True

	def _evaluate_regularizer(
		self, coef: NDArray[np.float64]
	) -> Tuple[float, NDArray[np.float64]]:
		"""Return regularization penalty and gradient.

		Supported callable outputs:
		- scalar penalty
		- (penalty, grad)
		- grad ndarray (same shape as coef)
		"""
		if self.regularizer is None:
			return 0.0, np.zeros_like(coef)

		result = self.regularizer(coef)
		if isinstance(result, tuple):
			if len(result) != 2:
				raise TrainingError("regularizer tuple output must be (penalty, grad)")
			penalty = float(result[0])
			grad = np.asarray(result[1], dtype=np.float64)
			if grad.shape != coef.shape:
				raise TrainingError("regularizer gradient has invalid shape")
			return penalty, grad

		arr = np.asarray(result)
		if arr.shape == coef.shape:
			return 0.0, arr.astype(np.float64)

		if np.isscalar(result):
			return float(result), np.zeros_like(coef)

		raise TrainingError("regularizer must return scalar, gradient array, or (penalty, grad)")

	def _apply_constraint(self, coef: NDArray[np.float64]) -> NDArray[np.float64]:
		if self.constraint is None:
			return coef

		constrained = self.constraint(coef)
		constrained = np.asarray(constrained, dtype=np.float64)
		if constrained.shape != coef.shape:
			raise TrainingError("constraint returned array with invalid shape")
		return constrained

	def _calculate_hessian(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Compute Hessian for linear least-squares objective: H = X^T X."""
		return X.T @ X

	def _calculate_gradient(
		self,
		X: NDArray[np.float64],
		coef: NDArray[np.float64],
		y: NDArray[np.float64],
	) -> NDArray[np.float64]:
		"""Compute multitask gradient: dL/dW = (XW^T - y)^T X."""
		pred = X @ coef.T
		return (pred - y).T @ X

	def _newton_update(
		self,
		coef: NDArray[np.float64],
		grad: NDArray[np.float64],
		hessian: NDArray[np.float64],
		lr: float,
		active_features: NDArray[np.bool_],
	) -> NDArray[np.float64]:
		"""Apply Newton step on active features and freeze inactive/ill-conditioned ones."""
		updated = coef.copy()
		if not np.any(active_features):
			return updated

		H_active = hessian[np.ix_(active_features, active_features)]
		if H_active.size == 0:
			return updated

		# Small damping improves stability for near-singular Hessian blocks.
		H_active = H_active + np.eye(H_active.shape[0], dtype=np.float64) * self._NEWTON_EPS

		for out_idx in range(updated.shape[0]):
			g_active = grad[out_idx, active_features]
			try:
				delta_active = np.linalg.solve(H_active, g_active)
			except np.linalg.LinAlgError:
				delta_active = np.linalg.pinv(H_active) @ g_active
			updated[out_idx, active_features] = (
				updated[out_idx, active_features] - lr * delta_active
			)

		return updated

	def initialize(self, data: NDArray[np.float64], lags: Optional[int] = None, **kwargs: Any) -> None:
		self._validate_components()

		X = np.asarray(data, dtype=np.float64)
		y_raw = kwargs.get("targets")
		if y_raw is None:
			raise TrainingError(
				"initialize requires precomputed targets via targets=<ndarray>. "
				"Lagged features should be prepared by LagEngine."
			)

		y = np.asarray(y_raw, dtype=np.float64)
		if X.ndim != 2:
			raise TrainingError("Expected 2D lagged feature matrix")
		if y.ndim == 1:
			y = y[:, np.newaxis]
		if y.ndim != 2:
			raise TrainingError("targets must be 1D or 2D array")
		if X.shape[0] != y.shape[0]:
			raise TrainingError("Lagged features and targets must have the same number of rows")

		self._n_features = X.shape[1]
		self._n_outputs = y.shape[1]
		self._variable_control_layer = _VariableControlLayer(self._n_features)
		self._variable_mask = self._variable_control_layer.mask
		self.coef_ = np.zeros(
			(self._n_outputs, self._n_features),
			dtype=np.float64,
		)
		self.intercept_ = np.zeros(
			self._n_outputs,
			dtype=np.float64,
		)
		self._X_train = X
		self._y_train = y
		self._loss_history = []
		self._optimizer_proxy = _OptimizerProxy(self.learning_rate)
		self._fitted = False

	def fit(
		self,
		X: Optional[NDArray[np.float64]] = None,
		y: Optional[NDArray[np.float64]] = None,
		**kwargs: Any,
	) -> Dict[str, Any]:
		"""Train using gradient descent and return BaseGrangerModel-compatible result."""
		if X is not None or y is not None:
			if X is None or y is None:
				raise TrainingError("Both X and y must be provided when using sklearn-style fit")
			lags = int(kwargs.get("lags", 1))
			self.initialize(X, lags=lags, targets=y)

		if self._X_train is None or self._y_train is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		X_train = self._X_train
		if self._variable_control_layer is not None:
			X_train = self._variable_control_layer.transform(X_train)

		y_train = self._y_train
		X_proc, y_proc, X_offset, y_offset, X_scale, _ = _preprocess_data(
			X_train,
			y_train,
			fit_intercept=self.fit_intercept,
			copy=True,
		)

		n_samples, n_features = X_proc.shape
		n_outputs = y_proc.shape[1]

		if n_samples == 0:
			raise TrainingError("Cannot train on an empty dataset.")

		coef = np.array(self.coef_, dtype=np.float64, copy=True)
		if coef.shape != (n_outputs, n_features):
			raise TrainingError(
				"Current coefficients do not match the training data."
			)

		self._fitted = False
		self.coef_ = coef.copy()
		self._update_analytic_intercept(X_offset, y_offset)

		# Reset training-loop learning rate, separately from callback reset.
		self._optimizer_proxy = _OptimizerProxy(self.learning_rate)

		self._loss_history = []
		callback_state: Dict[str, Any] = {
			"model": self,
			"optimizer": self._optimizer_proxy,
			"stop_training": False,
			"stop_reason": None,
			"loss_history": self._loss_history,
		}
		self._run_callback_hook("on_train_beginning", callback_state)

		for epoch in range(self.max_iter):
			callback_state["epoch"] = epoch
			if not self._run_callback_hook("on_epoch_beginning", callback_state):
				break

			lr = float(self._optimizer_proxy.param_groups[0]["lr"])
			if self.batch_size is None:
				batch_ranges = [(0, n_samples)]
			else:
				batch_ranges = [
					(start, min(start + self.batch_size, n_samples))
					for start in range(0, n_samples, self.batch_size)
				]

			for start, end in batch_ranges:
				Xb = X_proc[start:end, :]
				yb = y_proc[start:end, :]
				grad = self._calculate_gradient(Xb, coef, yb)
				hessian = self._calculate_hessian(Xb)

				h_diag = np.diag(hessian)
				mask_vec = (
					self._variable_control_layer.mask
					if self._variable_control_layer is not None
					else np.ones(n_features, dtype=np.float64)
				)
				# omit_variables zeroes feature columns, which zeroes corresponding Hessian
				# rows/cols; we freeze those coefficients using active_features mask.
				active_features = (mask_vec > 0.0) & (np.abs(h_diag) > self._NEWTON_EPS)

				_, reg_grad = self._evaluate_regularizer(coef)
				grad = grad + reg_grad
				coef = self._newton_update(coef, grad, hessian, lr, active_features)
				coef = self._apply_constraint(coef)

			self.coef_ = coef.copy()
			self._update_analytic_intercept(X_offset, y_offset)

			pred_full = X_proc @ self.coef_.T
			mse = float(np.mean((pred_full - y_proc) ** 2))
			reg_penalty, _ = self._evaluate_regularizer(coef)
			epoch_loss = float(mse + reg_penalty)
			self._loss_history.append(epoch_loss)

			callback_state["epoch_loss"] = epoch_loss
			callback_state["loss_history"] = self._loss_history
			if not self._run_callback_hook("on_epoch_end", callback_state):
				break

			if self.verbose:
				print(f"Epoch {epoch + 1}/{self.max_iter} - Loss: {epoch_loss:.8f}")

			if callback_state.get("stop_training"):
				break

			if len(self._loss_history) >= 2:
				if abs(self._loss_history[-2] - self._loss_history[-1]) <= self.tol:
					callback_state["stop_reason"] = callback_state.get("stop_reason") or "tolerance"
					break

		self.coef_ = coef
		self._run_callback_hook("on_train_end", callback_state)

		# Do not overwrite weights restored by an end-of-epoch callback.
		self._update_analytic_intercept(X_offset, y_offset)
		self._fitted = True

		self._run_callback_hook("on_train_end", callback_state)

		# Callbacks may restore a previous kernel.
		# In this backend the intercept is derived analytically from that kernel.
		self._update_analytic_intercept(X_offset, y_offset)

		forecasts = self.predict(self._X_train)

		reg_penalty, _ = self._evaluate_regularizer(self.coef_)
		final_loss = float(
			np.mean((forecasts - self._y_train) ** 2) + reg_penalty
		)
		return {
			"test_statistic": final_loss,
			"p_value": np.nan,
			"weights": self.get_weights(),
			"forecasts": forecasts,
			"history": {
				"loss": self._loss_history,
				"stop_reason": callback_state.get("stop_reason"),
			},
		}

	def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		if not self._fitted or not hasattr(self, "coef_"):
			raise ModelNotFittedError("Model is not fitted. Call fit(...) first.")

		x_arr = np.asarray(X, dtype=np.float64)
		if x_arr.ndim != 2:
			raise TrainingError("X must be a 2D array")
		if self._variable_control_layer is not None:
			x_arr = self._variable_control_layer.transform(x_arr)

		return x_arr @ self.coef_.T + self.intercept_

	def set_weights(
		self,
		weights: List[NDArray[np.float64]],
	) -> None:
		"""Assign coefficients without changing the input-control mask."""
		if self._n_features is None or self._n_outputs is None:
			raise ModelNotFittedError(
				"Model is not initialized. Call initialize(...) first."
			)

		if not isinstance(weights, list) or len(weights) != 2:
			raise TrainingError(
				"This linear model requires [kernel, bias]."
			)

		kernel = np.array(weights[0], dtype=np.float64, copy=True)
		bias = np.array(weights[1], dtype=np.float64, copy=True)

		expected_kernel = (self._n_features, self._n_outputs)
		expected_bias = (self._n_outputs,)

		if kernel.shape != expected_kernel:
			raise TrainingError(
				f"Kernel shape is {kernel.shape}, "
				f"expected {expected_kernel}."
			)
		if bias.shape != expected_bias:
			raise TrainingError(
				f"Bias shape is {bias.shape}, expected {expected_bias}."
			)
		if not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(bias)):
			raise TrainingError("Weights must contain finite values.")
		if not self.fit_intercept and np.any(bias != 0.0):
			raise TrainingError(
				"Non-zero bias is not allowed with fit_intercept=False."
			)

		self.coef_ = kernel.T.copy()
		self.intercept_ = bias.copy()

	def get_weights(self) -> List[NDArray[np.float64]]:
		"""Return coefficient copies: [kernel, bias].

		kernel: (n_features, n_outputs)
		bias:   (n_outputs,)
		Input-control state is excluded.
		"""
		if (
			self._n_features is None
			or self._n_outputs is None
			or not hasattr(self, "coef_")
			or not hasattr(self, "intercept_")
		):
			raise ModelNotFittedError(
				"Model is not initialized. Call initialize(...) first."
			)

		kernel = np.array(
			self.coef_.T,
			dtype=np.float64,
			copy=True,
		)
		bias = np.array(
			self.intercept_,
			dtype=np.float64,
			copy=True,
		).reshape(self._n_outputs)

		return [kernel, bias]

	def omit_variables(self, variable_indices: List[int]) -> None:
		if self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")
		if self._variable_control_layer is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		# Match PyTorch behavior: each call starts from fully enabled variables.
		self._variable_control_layer.reset()
		self._variable_control_layer.omit(variable_indices)
		self._variable_mask = self._variable_control_layer.mask

	def set_regularizer(self, regularizer: Any) -> None:
		self._validate_components()
		self.regularizer = regularizer

	def set_constraint(self, constraint: Any) -> None:
		self._validate_components()
		self.constraint = constraint
	
	def reset_callbacks(
		self,
		run_name: str,
		callback_updates: Optional[
			Mapping[Union[int, str, type], Mapping[str, Any]]
		] = None,
	) -> None:
		fresh_callbacks = reset_callback_list(
			self.callbacks,
			run_name,
			callback_updates,
			expected_type=Callback,
			clone_callback=clone_custom_callback,
		)
		self.callbacks = fresh_callbacks

	def _update_analytic_intercept(
		self,
		X_offset: NDArray[np.float64],
		y_offset: NDArray[np.float64],
	) -> None:
		"""Update the intercept for current coefficients and masked inputs."""
		if self.fit_intercept:
			bias = (
				np.asarray(y_offset, dtype=np.float64)
				- np.asarray(X_offset, dtype=np.float64) @ self.coef_.T
			)
			self.intercept_ = np.asarray(
				bias, dtype=np.float64
			).reshape(self._n_outputs).copy()
		else:
			self.intercept_ = np.zeros(
				self._n_outputs,
				dtype=np.float64,
			)