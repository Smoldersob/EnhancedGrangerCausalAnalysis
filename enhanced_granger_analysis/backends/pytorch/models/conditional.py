
from __future__ import annotations

import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from typing import Any, Dict, List, Mapping, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

from .registry import register_model
from .network_parts import TorchFeatureMask

from ..callbacks.base import Callback
from ....core.components.callback_lifecycle import (
    clone_custom_callback,
    reset_callback_list,
)
from ....core.exceptions import (
	BackendNotAvailableError,
	ConstraintConfigurationError,
	RegularizerConfigurationError,
	BackendCompatibilityError,
	ModelNotFittedError,
	TrainingError,
)



@register_model(
    "conditional",
    "conditional_linear",
)
class PyTorchGrangerModel:
	"""PyTorch implementation of Granger model with variable masking and linear coefficients."""

	def __init__(
		self,
		backend: str = "pytorch",
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		optimizer: Optional[Any] = None,
		loss: Optional[Any] = None,
		callbacks: List[Callback] = None,
		learning_rate: float = 1e-3,
		gradient_accumulation_steps: int = 1,
		loss_fn: Optional[Any] = None,
		epochs: int = 100,
		batch_size: Optional[int] = 32,
		verbose: int = 0,
		device: Optional[str] = None,
		**kwargs: Any
	):
		super().__init__(
			backend=backend,
			regularizer=regularizer,
			constraint=constraint,
			needs_reinit=False,
		)

		if torch is None:
			raise BackendNotAvailableError(
				"PyTorch is required to use PyTorchGrangerModel. Install torch first."
			)

		self.learning_rate = learning_rate
		self.gradient_accumulation_steps = int(gradient_accumulation_steps)
		if self.gradient_accumulation_steps < 1:
			raise ConstraintConfigurationError("gradient_accumulation_steps must be >= 1")
		self.epochs = epochs
		self.batch_size = batch_size
		self.verbose = verbose
		
		self._optimizer_spec = optimizer
		self._loss_spec = loss if loss is not None else loss_fn
		self.callbacks = callbacks or []

		self.optimizer_cls = self._resolve_optimizer(self._optimizer_spec)
		self.loss_fn = self._resolve_loss(self._loss_spec)
		self.device = self._resolve_device(device)

		self.model: Optional[Any] = None
		self._variable_control_layer: Optional[Any] = None
		self._coefficient_layer: Optional[Any] = None
		self._optimizer: Optional[Any] = None

		self._n_features: Optional[int] = None
		self._n_outputs: Optional[int] = None
		self._variable_mask: Optional[NDArray[np.float64]] = None
		self._X_train: Optional[NDArray[np.float64]] = None
		self._y_train: Optional[NDArray[np.float64]] = None
		self._loss_history: List[float] = []

		self._validate_torch_components()
		self._validate_callbacks()

	def _resolve_device(self, device: Optional[str]) -> Any:
		"""Normalize requested device alias and enforce CPU/GPU selection."""
		if device is None or str(device).strip().lower() in {"", "auto"}:
			return torch.device("cuda" if torch.cuda.is_available() else "cpu")

		normalized = str(device).strip().lower()
		if normalized in {"cpu", "cpu-only"}:
			return torch.device("cpu")
		if normalized in {"gpu", "cuda"}:
			if not torch.cuda.is_available():
				raise BackendNotAvailableError(
					"CUDA was requested for PyTorch but no CUDA device is available."
				)
			return torch.device("cuda")
		if normalized.startswith("cuda"):
			if not torch.cuda.is_available():
				raise BackendNotAvailableError(
					f"Requested CUDA device '{normalized}' is unavailable on this machine."
				)
			return torch.device(normalized)
		return torch.device(normalized)

	def _resolve_optimizer(self, optimizer: Optional[Union[str, Type[Any], Any]]) -> Any:
		"""Resolve optimizer spec to optimizer class."""
		if optimizer is None:
			return torch.optim.Adam

		if isinstance(optimizer, str):
			optimizers = {
				"adam": torch.optim.Adam,
				"sgd": torch.optim.SGD,
				"rmsprop": torch.optim.RMSprop,
			}
			resolved = optimizers.get(optimizer.lower())
			if resolved is None:
				raise BackendCompatibilityError(
					f"Unsupported optimizer '{optimizer}'. Use one of: {sorted(optimizers.keys())}"
				)
			return resolved

		if isinstance(optimizer, type):
			if issubclass(optimizer, torch.optim.Optimizer):
				return optimizer
			raise BackendCompatibilityError(
				"optimizer class must inherit from torch.optim.Optimizer"
			)

		if callable(optimizer):
			return optimizer

		raise ConstraintConfigurationError(
			"optimizer must be string, optimizer class, or callable"
		)

	def _build_optimizer(self) -> Any:
		"""Create a fresh optimizer instance (state reset)."""
		if self._coefficient_layer is None:
			raise ModelNotFittedError(
				"Model layers are not initialized. Call initialize(...) first."
			)

		params = self._coefficient_layer.parameters()
		provider = self.optimizer_cls

		if isinstance(provider, type) and issubclass(provider, torch.optim.Optimizer):
			return provider(params, lr=self.learning_rate)

		if callable(provider):
			try:
				opt = provider(params, lr=self.learning_rate)
			except TypeError:
				opt = provider(params)

			if isinstance(opt, torch.optim.Optimizer):
				return opt

			raise ConstraintConfigurationError(
				"optimizer callable must return torch.optim.Optimizer"
			)

		raise ConstraintConfigurationError(
			"Unable to build optimizer from provided optimizer specification"
		)

	def _resolve_loss(self, loss: Any) -> Any:
		"""Resolve loss spec to callable loss function."""
		if loss is None:
			return nn.MSELoss()

		if isinstance(loss, str):
			losses = {
				"mse": nn.MSELoss,
				"mae": nn.L1Loss,
				"l1": nn.L1Loss,
				"huber": nn.SmoothL1Loss,
			}
			loss_cls = losses.get(loss.lower())
			if loss_cls is None:
				raise ConstraintConfigurationError(
					f"Unsupported loss '{loss}'. Use one of: {sorted(losses.keys())}"
				)
			return loss_cls()

		if isinstance(loss, nn.Module) or callable(loss):
			return loss

		raise ConstraintConfigurationError(
			"loss must be string, torch.nn.Module, or callable"
		)

	def _validate_callbacks(self) -> None:
		"""Validate callback list passed to model initializer."""
		if not isinstance(self.callbacks, list):
			raise ConstraintConfigurationError("callbacks must be a list of Callback objects")

		for cb in self.callbacks:
			if not isinstance(cb, Callback):
				raise ConstraintConfigurationError(
					"All PyTorch callbacks must inherit from callbacks.base_callback.Callback"
				)

	def _run_callback_hook(self, hook_name: str, state: Dict[str, Any]) -> bool:
		"""Run hook across callbacks and return whether training should continue."""
		for callback in self.callbacks:
			hook = getattr(callback, hook_name)
			response = hook(state)
			if response is False:
				state["stop_training"] = True
				if not state.get("stop_reason"):
					state["stop_reason"] = callback.__class__.__name__ + ':' + hook_name
				return False
		return True

	def _validate_torch_components(self) -> None:
		"""Validate optional regularizer/constraint for PyTorch backend."""
		if self.regularizer is not None:
			from ..regularizers.base import BasePyTorchRegularizer
			
			if not isinstance(self.regularizer, BasePyTorchRegularizer):
				raise RegularizerConfigurationError(
					"regularizer must be None or an instance of "
					"BasePyTorchRegularizer."
				)

		if self.constraint is not None and not callable(self.constraint):
			raise ConstraintConfigurationError(
				"constraint must be callable for PyTorchGrangerModel"
			)

	def initialize(
		self,
		data: NDArray[np.float64],
		lags: Optional[int] = None,
		**kwargs: Any,
	) -> None:
		"""Initialize model using lagged features prepared externally (e.g. LagEngine)."""
		self._validate_torch_components()

		x = np.asarray(data, dtype=np.float64)
		y_raw = kwargs.get("targets")
		if y_raw is None:
			raise TrainingError(
				"initialize requires precomputed targets via targets=<ndarray>. "
				"Lagged features should be prepared by LagEngine."
			)

		y = np.asarray(y_raw, dtype=np.float64)
		if x.ndim != 2:
			raise TrainingError("Expected 2D lagged feature matrix with shape (n_samples, n_lagged_features)")
		if y.ndim == 1:
			y = y[:, np.newaxis]
		if y.ndim != 2:
			raise TrainingError("targets must be 1D or 2D array")
		if x.shape[0] != y.shape[0]:
			raise TrainingError("Lagged features and targets must have the same number of rows")

		n_features = x.shape[1]
		n_outputs = y.shape[1]

		variable_control_layer = TorchFeatureMask(n_features)

		coefficient_layer = nn.Linear(n_features, n_outputs, bias=True)

		self.model = nn.Sequential(variable_control_layer, coefficient_layer).to(self.device)
		self._variable_control_layer = variable_control_layer
		self._coefficient_layer = coefficient_layer
		try:
			self._optimizer = self._build_optimizer()
		except Exception as exc:
			raise TrainingError(f"Failed to initialize optimizer: {exc}") from exc

		self._n_features = n_features
		self._n_outputs = n_outputs
		self._variable_mask = np.ones(n_features, dtype=np.float64)
		self._X_train = x
		self._y_train = y
		self._loss_history = []
		self._fitted = False

	def _regularization_penalty(self) -> Any:
		"""Compute optional regularization penalty for coefficient weights."""
		if self.regularizer is None or self._coefficient_layer is None:
			return torch.tensor(0.0, device=self.device)

		weights = self._coefficient_layer.weight
		penalty = self.regularizer(weights)
		if not torch.is_tensor(penalty):
			penalty = torch.tensor(float(penalty), device=self.device)
		return penalty

	def _apply_constraint(self) -> None:
		"""Apply optional post-step constraint on coefficient weights."""
		if self.constraint is None or self._coefficient_layer is None:
			return

		with torch.no_grad():
			constrained = self.constraint(self._coefficient_layer.weight.data)
			if not torch.is_tensor(constrained):
				raise TrainingError("constraint must return torch.Tensor")
			if constrained.shape != self._coefficient_layer.weight.data.shape:
				raise TrainingError("constraint returned tensor with invalid shape")
			self._coefficient_layer.weight.data.copy_(constrained)

	def fit(self) -> Dict[str, Any]:
		"""Fit model using a PyTorch training loop and return standardized results."""
		if self.model is None or self._X_train is None or self._y_train is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")
		if self._coefficient_layer is None:
			raise ModelNotFittedError("Layers are not initialized. Call initialize(...) first.")

		# Always recreate optimizer to reset internal moments/state for each new fit.
		self._optimizer = self._build_optimizer()

		x_tensor = torch.tensor(self._X_train, dtype=torch.float32, device=self.device)
		y_tensor = torch.tensor(self._y_train, dtype=torch.float32, device=self.device)

		if self.batch_size is None:
			loader = [(x_tensor, y_tensor)]
		else:
			dataset = TensorDataset(x_tensor, y_tensor)
			loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		self.model.train()
		self._loss_history = []
		callback_state: Dict[str, Any] = {
			"model": self,
			"optimizer": self._optimizer,
			"stop_training": False,
			"stop_reason": None,
			"loss_history": self._loss_history,
		}
		self._run_callback_hook("on_train_beginning", callback_state)

		uses_closure = isinstance(self._optimizer, torch.optim.LBFGS)
		if uses_closure and self.gradient_accumulation_steps > 1:
			raise ConstraintConfigurationError(
				"gradient_accumulation_steps > 1 is not supported with closure-based optimizer stepping"
			)

		for epoch in range(self.epochs):
			callback_state["epoch"] = epoch
			if not self._run_callback_hook("on_epoch_beginning", callback_state):
				break

			epoch_losses: List[float] = []
			self._optimizer.zero_grad()
			for batch_idx, (xb, yb) in enumerate(loader):
				if uses_closure:
					def _default_closure() -> Any:
						self._optimizer.zero_grad()
						pred_local = self.model(xb)
						loss_local = self.loss_fn(pred_local, yb)
						loss_local = loss_local + self._regularization_penalty()
						loss_local.backward()
						return loss_local

					closure_fn = _default_closure

					step_result = self._optimizer.step(closure_fn)
					self._apply_constraint()

					if torch.is_tensor(step_result):
						epoch_losses.append(float(step_result.detach().cpu().item()))
					else:
						with torch.no_grad():
							loss_value = self.loss_fn(self.model(xb), yb) + self._regularization_penalty()
						epoch_losses.append(float(loss_value.detach().cpu().item()))
					continue
				else:
					pred = self.model(xb)
					loss = self.loss_fn(pred, yb)
					loss = loss + self._regularization_penalty()
					loss = loss / self.gradient_accumulation_steps
					loss.backward()

					if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(loader) - 1:
						self._optimizer.step()
						self._optimizer.zero_grad()
						self._apply_constraint()
						epoch_losses.append(float(loss.detach().cpu().item() * self.gradient_accumulation_steps))

			epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
			self._loss_history.append(epoch_loss)
			callback_state["epoch_loss"] = epoch_loss
			callback_state["loss_history"] = self._loss_history
			if not self._run_callback_hook("on_epoch_end", callback_state):
				break

			if self.verbose:
				print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.8f}")

			if callback_state.get("stop_training"):
				break

		self._run_callback_hook("on_train_end", callback_state)

		self.model.eval()
		with torch.no_grad():
			forecasts_t = self.model(x_tensor)
		forecasts = forecasts_t.detach().cpu().numpy()

		self._fitted = True
		final_loss = self._loss_history[-1] if self._loss_history else float("nan")

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
		"""Generate predictions from fitted PyTorch model."""
		if not self._fitted or self.model is None or self._n_features is None:
			raise ModelNotFittedError("Model is not fitted. Call fit(...) first.")

		x_arr = np.asarray(X, dtype=np.float64)
		if x_arr.ndim != 2:
			raise TrainingError("X must be a 2D array")
		if x_arr.shape[1] != self._n_features:
			raise TrainingError(
				f"X has {x_arr.shape[1]} features, expected {self._n_features}"
			)

		x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=self.device)
		self.model.eval()
		with torch.no_grad():
			pred = self.model(x_tensor)
		return pred.detach().cpu().numpy().astype(np.float64)

	def set_weights(
		self,
		weights: List[NDArray[np.float64]],
	) -> None:
		if self._coefficient_layer is None:
			raise ModelNotFittedError(
				"Model is not initialized. Call initialize(...) first."
			)

		layer = self._coefficient_layer

		try:
			kernel, bias = self._validated_weight_arrays(
				weights,
				[
					(self._n_features, self._n_outputs),
					(self._n_outputs,),
				],
			)
		except (TypeError, ValueError) as exc:
			raise TrainingError(str(exc)) from exc

		kernel_tensor = torch.tensor(
			kernel.T,
			dtype=layer.weight.dtype,
			device=layer.weight.device,
		)
		bias_tensor = torch.tensor(
			bias,
			dtype=layer.bias.dtype,
			device=layer.bias.device,
		)

		if (
			not torch.isfinite(kernel_tensor).all().item()
			or not torch.isfinite(bias_tensor).all().item()
		):
			raise TrainingError(
				"Weights are not finite in the model's numeric dtype."
			)

		with torch.no_grad():
			layer.weight.copy_(kernel_tensor)
			layer.bias.copy_(bias_tensor)

	def get_weights(self) -> List[NDArray[np.float64]]:
		if self._coefficient_layer is None:
			raise ModelNotFittedError(
				"Model is not initialized. Call initialize(...) first."
			)

		layer = self._coefficient_layer

		kernel = np.array(
			layer.weight.detach().cpu().numpy().T,
			dtype=np.float64,
			copy=True,
		)
		bias = np.array(
			layer.bias.detach().cpu().numpy(),
			dtype=np.float64,
			copy=True,
		)

		return [kernel, bias]
	
	def omit_variables(self, variable_indices: List[int]) -> None:
		"""Replace omitted input features without changing coefficients."""
		if (
			self._variable_control_layer is None
			or self._n_features is None
		):
			raise ModelNotFittedError(
				"Model is not initialized. Call initialize(...) first."
			)

		indices = np.asarray(variable_indices)

		if indices.ndim != 1:
			raise TrainingError(
				"variable_indices must be a one-dimensional list of integers."
			)

		if indices.size:
			if indices.dtype.kind not in {"i", "u"}:
				raise TrainingError(
					"Variable indices must be integers, not floats or booleans."
				)

			if (
				np.any(indices < 0)
				or np.any(indices >= self._n_features)
			):
				raise TrainingError(
					f"Variable indices must belong to "
					f"[0, {self._n_features - 1}]."
				)

		indices = indices.astype(np.intp, copy=False)

		variable_mask = np.ones(
			self._n_features,
			dtype=np.float64,
		)
		variable_mask[indices] = 0.0

		mask_buffer = self._variable_control_layer.mask

		with torch.no_grad():
			mask_buffer.copy_(
				torch.as_tensor(
					variable_mask,
					dtype=mask_buffer.dtype,
					device=mask_buffer.device,
				)
			)

		self._variable_mask = variable_mask

	def set_regularizer(self, regularizer: Any) -> None:
		"""Set regularizer with PyTorch component validation."""
		self._validate_torch_components()
		self.regularizer = regularizer
		

	def set_constraint(self, constraint: Any) -> None:
		"""Set constraint with PyTorch component validation."""
		self._validate_torch_components()
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