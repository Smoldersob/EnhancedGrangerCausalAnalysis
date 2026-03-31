from __future__ import annotations

import importlib
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

from .base_model import BaseGrangerModel
from ...callbacks.base_callback import Callback
from ...core.exceptions import (
	BackendNotAvailableError,
	ConstraintConfigurationError,
	ModelNotFittedError,
	TrainingError,
)

if find_spec("torch") is not None:
	torch = importlib.import_module("torch")
	nn = importlib.import_module("torch.nn")
	TensorDataset = importlib.import_module("torch.utils.data").TensorDataset
	DataLoader = importlib.import_module("torch.utils.data").DataLoader
else:  # pragma: no cover - runtime dependency check
	torch = None
	nn = None
	TensorDataset = None
	DataLoader = None


class PyTorchGrangerModel(BaseGrangerModel):
	"""PyTorch implementation of Granger model with variable masking and linear coefficients."""

	def __init__(
		self,
		backend: str = "pytorch",
		scaler: Optional[Any] = None,
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		optimizer: Optional[Union[str, Type[Any], Any]] = None,
		loss: Optional[Union[str, Any]] = None,
		callbacks: Optional[List[Callback]] = None,
		optimizer_cls: Optional[Any] = None,
		learning_rate: float = 1e-3,
		loss_fn: Optional[Any] = None,
		epochs: int = 100,
		batch_size: Optional[int] = 32,
		verbose: int = 0,
		device: Optional[str] = None,
	) -> None:
		super().__init__(
			backend=backend,
			scaler=scaler,
			regularizer=regularizer,
			constraint=constraint,
		)

		if torch is None:
			raise BackendNotAvailableError(
				"PyTorch is required to use PyTorchGrangerModel. Install torch first."
			)

		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.verbose = verbose

		self._optimizer_spec = optimizer if optimizer is not None else optimizer_cls
		self._loss_spec = loss if loss is not None else loss_fn
		self.callbacks = callbacks or []

		self.optimizer_cls = self._resolve_optimizer(self._optimizer_spec)
		self.loss_fn = self._resolve_loss(self._loss_spec)

		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)

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
				raise ConstraintConfigurationError(
					f"Unsupported optimizer '{optimizer}'. Use one of: {sorted(optimizers.keys())}"
				)
			return resolved

		if isinstance(optimizer, type):
			if issubclass(optimizer, torch.optim.Optimizer):
				return optimizer
			raise ConstraintConfigurationError(
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

	def _resolve_loss(self, loss: Optional[Union[str, Any]]) -> Any:
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
			is_valid_reg = isinstance(self.regularizer, nn.Module) or callable(self.regularizer)
			if not is_valid_reg:
				raise ConstraintConfigurationError(
					"regularizer must be torch.nn.Module or callable for PyTorchGrangerModel"
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

		variable_control_layer = nn.Linear(n_features, n_features, bias=False)
		with torch.no_grad():
			variable_control_layer.weight.copy_(torch.eye(n_features, dtype=torch.float32))
		variable_control_layer.weight.requires_grad_(False)

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
		self._X_train = X
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

		X_tensor = torch.tensor(self._X_train, dtype=torch.float32, device=self.device)
		y_tensor = torch.tensor(self._y_train, dtype=torch.float32, device=self.device)

		if self.batch_size is None:
			loader = [(X_tensor, y_tensor)]
		else:
			dataset = TensorDataset(X_tensor, y_tensor)
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

		for epoch in range(self.epochs):
			callback_state["epoch"] = epoch
			if not self._run_callback_hook("on_epoch_beginning", callback_state):
				break

			epoch_losses: List[float] = []
			for xb, yb in loader:
				self._optimizer.zero_grad()
				pred = self.model(xb)
				loss = self.loss_fn(pred, yb)
				loss = loss + self._regularization_penalty()
				loss.backward()
				self._optimizer.step()
				self._apply_constraint()
				epoch_losses.append(float(loss.detach().cpu().item()))

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
			forecasts_t = self.model(X_tensor)
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

		X_arr = np.asarray(X, dtype=np.float64)
		if X_arr.ndim != 2:
			raise TrainingError("X must be a 2D array")
		if X_arr.shape[1] != self._n_features:
			raise TrainingError(
				f"X has {X_arr.shape[1]} features, expected {self._n_features}"
			)

		X_tensor = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
		self.model.eval()
		with torch.no_grad():
			pred = self.model(X_tensor)
		return pred.detach().cpu().numpy().astype(np.float64)

	def set_weights(
		self, weights: Union[NDArray[np.float64], List[NDArray[np.float64]]]
	) -> None:
		"""Set coefficient-layer weights (kernel or kernel+bias)."""
		if self._coefficient_layer is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		with torch.no_grad():
			if isinstance(weights, list):
				if len(weights) == 1:
					w = torch.tensor(weights[0], dtype=torch.float32, device=self.device)
					self._coefficient_layer.weight.copy_(w.T if w.shape == self._coefficient_layer.weight.T.shape else w)
				elif len(weights) == 2:
					w = torch.tensor(weights[0], dtype=torch.float32, device=self.device)
					b = torch.tensor(weights[1], dtype=torch.float32, device=self.device)
					self._coefficient_layer.weight.copy_(w.T if w.shape == self._coefficient_layer.weight.T.shape else w)
					self._coefficient_layer.bias.copy_(b)
				else:
					raise TrainingError("weights list must contain kernel or [kernel, bias]")
			else:
				w = torch.tensor(weights, dtype=torch.float32, device=self.device)
				self._coefficient_layer.weight.copy_(w.T if w.shape == self._coefficient_layer.weight.T.shape else w)

	def get_weights(self) -> List[NDArray[np.float64]]:
		"""Return coefficient-layer kernel in a one-element list."""
		if self._coefficient_layer is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		kernel = self._coefficient_layer.weight.detach().cpu().numpy().T
		return [np.asarray(kernel, dtype=np.float64)]

	def omit_variables(self, variable_indices: List[int]) -> None:
		"""Set selected variable mask entries to zero in non-trainable diagonal layer."""
		if self._variable_control_layer is None or self._n_features is None:
			raise ModelNotFittedError("Model is not initialized. Call initialize(...) first.")

		if self._variable_mask is None:
			self._variable_mask = np.ones(self._n_features, dtype=np.float64)

		for idx in variable_indices:
			if idx < 0 or idx >= self._n_features:
				raise TrainingError(f"Variable index {idx} out of range [0, {self._n_features - 1}]")
			self._variable_mask[idx] = 0.0

		diagonal_kernel = np.diag(self._variable_mask).astype(np.float32)
		with torch.no_grad():
			self._variable_control_layer.weight.copy_(
				torch.tensor(diagonal_kernel, dtype=torch.float32, device=self.device)
			)

	def set_regularizer(self, regularizer: Any) -> None:
		"""Set regularizer with PyTorch component validation."""
		self.regularizer = regularizer
		self._validate_torch_components()

	def set_constraint(self, constraint: Any) -> None:
		"""Set constraint with PyTorch component validation."""
		self.constraint = constraint
		self._validate_torch_components()

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
			"message": "PyTorchGrangerModel nie posiada parametrów do hiperoptymalizacji.",
		}
