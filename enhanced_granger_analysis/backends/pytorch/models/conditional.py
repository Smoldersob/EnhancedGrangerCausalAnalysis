from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from ....core.exceptions import (
	ModelNotFittedError,
	TrainingError,
)
from ....core.models.protocol import (
	CallbackUpdates,
	FloatArray,
	ModelWeights,
	TrainingResult,
)
from ....core.models.input_spec import ModelInputSpec
from ....core.models.validation import validate_training_arrays

from ..device import DeviceManager
from ..callbacks.base import Callback
from ..callbacks.reset import reset_callbacks_for_run
from ..constraints.base import BasePyTorchConstraint
from ..optimizers.adapter import PyTorchOptimizerAdapter
from ..regularizers.base import BasePyTorchRegularizer

from .network_parts import LinearGrangerNetwork
from .registry import register_model


@register_model("conditional", "conditional_linear")
class PyTorchConditionalAdapter:
	"""Granger protocol adapter for a separate linear nn.Module.

	Mirrors the Keras adapter's lifecycle. initialize() creates a predictor
	and an independent optimizer. fit() never resets weights or input masks.
	Callback hooks keep the names used by the existing PyTorch backend.
	"""

	def __init__(
		self,
		*,
		input_spec: ModelInputSpec,
		optimizer: PyTorchOptimizerAdapter,
		loss: Any = None,
		regularizer: BasePyTorchRegularizer | None = None,
		constraint: BasePyTorchConstraint | None = None,
		callbacks: Sequence[Callback] | None = None,
		epochs: int = 100,
		batch_size: int | None = 32,
		verbose: int = 0,
		device: str | None = None,
		use_bias: bool = True,
		gradient_accumulation_steps: int = 1,
		seed: int | None = None,
		**kwargs,
	):
		if not isinstance(optimizer, PyTorchOptimizerAdapter):
			raise TypeError("optimizer must be a prepared PyTorchOptimizerAdapter.")
		if loss is None:
			loss = nn.MSELoss()
		if isinstance(loss, type) or not callable(loss):
			raise TypeError("loss must be a ready loss instance or function.")
		if regularizer is not None and not isinstance(regularizer, BasePyTorchRegularizer):
			raise TypeError("regularizer must be a BasePyTorchRegularizer or None.")
		if constraint is not None and not isinstance(constraint, BasePyTorchConstraint):
			raise TypeError("constraint must be a BasePyTorchConstraint or None.")
		if not isinstance(use_bias, bool):
			raise TypeError("use_bias must be a bool.")
		self.callbacks = list(callbacks) if callbacks is not None else []
		if any(not isinstance(callback, Callback) for callback in self.callbacks):
			raise TypeError("All callbacks must inherit from PyTorch Callback.")

		self.loss = loss
		self.regularizer = regularizer
		self.constraint = constraint
		self.epochs = epochs
		self.batch_size = batch_size
		self.verbose = verbose
		self.use_bias = use_bias
		self.gradient_accumulation_steps = gradient_accumulation_steps
		self.device_manager = DeviceManager(
			requested_device=device,
		)

		# Keep configuration only, not the caller's active native optimizer.
		self._optimizer_template = PyTorchOptimizerAdapter(
			optimizer.optimizer_class, options=optimizer.options,
		)
		self._optimizer_adapter: PyTorchOptimizerAdapter | None = None
		self._runtime_loss: Any = None
		self.model: LinearGrangerNetwork | None = None
		self._n_features: int | None = None
		self._n_outputs: int | None = None
		self._fitted = False
		self._run_counter = 0
		self._loss_history: list[float] = []

		if not isinstance(input_spec, ModelInputSpec):
			raise TypeError("input_spec must be a ModelInputSpec.")

		self._input_spec = input_spec

		self._n_features = input_spec.n_features
		self._n_outputs = input_spec.n_outputs

		self._validate_training_config()
		self.rebuild(seed=seed)


	@property
	def backend(self) -> str:
		return "pytorch"

	@property
	def device(self) -> torch.device:
		return self.device_manager.device

	@property
	def input_spec(self) -> ModelInputSpec:
		return self._input_spec

	@property
	def is_fitted(self) -> bool:
		return self._fitted

	@property
	def optimizer(self) -> torch.optim.Optimizer:
		self._require_model()
		if self._optimizer_adapter is None:
			raise TrainingError("Optimizer is unavailable.")
		return self._optimizer_adapter.native

	# Compatibility views; the active objects have a single owner.
	@property
	def _optimizer(self) -> torch.optim.Optimizer:
		return self.optimizer

	@property
	def _coefficient_layer(self) -> nn.Linear:
		return self._require_model().coefficient_layer

	@property
	def _variable_control_layer(self) -> nn.Module:
		return self._require_model().variable_control_layer

	@staticmethod
	def _positive_int(value: Any, name: str) -> int:
		if (
			isinstance(value, (bool, np.bool_))
			or not isinstance(value, (int, np.integer))
			or value <= 0
		):
			raise ValueError(f"{name} must be a positive integer.")
		return int(value)

	def _validate_training_config(self) -> None:
		self._positive_int(self.epochs, "epochs")
		self._positive_int(self.gradient_accumulation_steps, "gradient_accumulation_steps")
		if self.batch_size is not None:
			self._positive_int(self.batch_size, "batch_size")
		if (
			isinstance(self.verbose, (bool, np.bool_))
			or not isinstance(self.verbose, (int, np.integer))
			or self.verbose not in (0, 1, 2)
		):
			raise ValueError("verbose must be 0, 1 or 2.")
		if self.gradient_accumulation_steps > 1:
			if issubclass(self._optimizer_template.optimizer_class, torch.optim.LBFGS):
				raise ValueError("LBFGS requires gradient_accumulation_steps=1.")
			loss = self._runtime_loss if self._runtime_loss is not None else self.loss
			if getattr(loss, "reduction", "mean") != "mean":
				raise ValueError("Gradient accumulation requires a mean-reduced loss.")

	def _require_model(self) -> LinearGrangerNetwork:
		if self.model is None:
			raise RuntimeError(
				"Predictor construction did not complete."
			)
		return self.model

	@staticmethod
	def _data_array(value: Any, name: str) -> np.ndarray:
		raw = np.asarray(value)
		if raw.ndim != 2:
			raise ValueError(f"{name} must be a two-dimensional array.")
		if raw.dtype.kind not in "fiu":
			raise TypeError(f"{name} must contain real numeric values.")
		with np.errstate(over="ignore", invalid="ignore"):
			array = np.array(raw, dtype=np.float32, copy=True)
		if not np.isfinite(array).all():
			raise ValueError(f"{name} must contain finite float32-representable values.")
		return array

	def rebuild(
		self,
		*,
		seed: int | None = None,
	) -> None:
		self._validate_training_config()

		if seed is not None:
			if (
				isinstance(seed, (bool, np.bool_))
				or not isinstance(seed, (int, np.integer))
				or not 0 <= seed <= np.iinfo(np.uint64).max
			):
				raise ValueError(
					"seed must be an integer in [0, 2**64 - 1] or None."
				)

			torch.manual_seed(int(seed))

		regularizer = None
		if self.regularizer is not None:
			regularizer = type(self.regularizer).from_config(
				self.regularizer.get_config()
			)

		network = LinearGrangerNetwork(
			n_features=self.input_spec.n_features,
			n_outputs=self.input_spec.n_outputs,
			constraint=deepcopy(self.constraint),
			regularizer=regularizer,
			use_bias=self.use_bias,
		).to(self.device)

		runtime_loss = (
			deepcopy(self.loss)
			if isinstance(self.loss, nn.Module)
			else self.loss
		)
		if isinstance(runtime_loss, nn.Module):
			runtime_loss = runtime_loss.to(self.device)

		optimizer_adapter = PyTorchOptimizerAdapter(
			self._optimizer_template.optimizer_class,
			options=self._optimizer_template.options,
		)
		optimizer_adapter.initialize(
			network.coefficient_layer.parameters()
		)

		self.model = network
		self._runtime_loss = runtime_loss
		self._optimizer_adapter = optimizer_adapter

		self._loss_history = []
		self._fitted = False

	def reset_components(
		self,
		*,
		reset_optimizer: bool = True,
		reset_callbacks: bool = True,
		run_name: str | None = None,
		callback_updates: CallbackUpdates | None = None,
	) -> None:
		self._require_model()
		if not isinstance(reset_optimizer, bool):
			raise TypeError("reset_optimizer must be a bool.")
		if not isinstance(reset_callbacks, bool):
			raise TypeError("reset_callbacks must be a bool.")
		if not reset_callbacks and callback_updates is not None:
			raise ValueError("callback_updates requires reset_callbacks=True.")
		next_counter = self._run_counter + 1
		name = run_name if run_name is not None else f"run_{next_counter:04d}"
		fresh_callbacks = None
		if reset_callbacks:
			fresh_callbacks = reset_callbacks_for_run(
				self.callbacks, run_name=name, callback_updates=callback_updates,
			)
		if reset_optimizer:
			if self._optimizer_adapter is None:
				raise TrainingError("Initial optimizer state is unavailable.")
			self._optimizer_adapter.reset()
		if fresh_callbacks is not None:
			self.callbacks = fresh_callbacks
			self._run_counter = next_counter

	def reset_callbacks(
		self,
		run_name: str,
		callback_updates: CallbackUpdates | None = None,
	) -> None:
		self.reset_components(
			reset_optimizer=False, reset_callbacks=True,
			run_name=run_name, callback_updates=callback_updates,
		)

	def _run_callback_hook(self, name: str, state: dict[str, Any]) -> None:
		for callback in self.callbacks:
			response = getattr(callback, name)(state)
			if response is False:
				state["stop_training"] = True
				if state.get("stop_reason") is None:
					state["stop_reason"] = f"{type(callback).__name__}:{name}"
		# Do not skip later callbacks just because an earlier one requested stop.

	def _objective(self, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
		network = self._require_model()
		loss = self._runtime_loss(network(xb), yb)
		if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
			raise ValueError("Loss must return a scalar torch.Tensor.")
		loss = loss + network.regularization_loss()
		if not torch.isfinite(loss).item():
			raise ValueError("Training objective is non-finite.")
		return loss

	def _train_epoch(
		self,
		x_train: torch.Tensor,
		y_train: torch.Tensor,
		batch_size: int,
	) -> float:
		network = self._require_model()
		optimizer = self.optimizer

		n_samples = int(x_train.shape[0])
		group_size = (
			batch_size * int(self.gradient_accumulation_steps)
		)
		uses_closure = isinstance(optimizer, torch.optim.LBFGS)
		weighted_loss = 0.0

		# Sequential slices give the same no-shuffle ordering as Keras.
		for group_start in range(0, n_samples, group_size):
			group_end = min(group_start + group_size, n_samples)
			optimizer.zero_grad(set_to_none=True)
			for start in range(group_start, group_end, batch_size):
				end = min(start + batch_size, group_end)
				xb = x_train[start:end]
				yb = y_train[start:end]
				if uses_closure:
					# LBFGS is restricted to one batch per accumulation group.
					def closure() -> torch.Tensor:
						optimizer.zero_grad(set_to_none=True)
						value = self._objective(xb, yb)
						value.backward()
						return value

					loss = optimizer.step(closure)
					if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
						raise TypeError("LBFGS must return a scalar loss Tensor.")
				else:
					loss = self._objective(xb, yb)
					# Correct also for a short final batch/group. Assumes a
					# mean-reduced data loss; regularizer gets total weight 1.
					fraction = (end - start) / (group_end - group_start)
					(loss * fraction).backward()
				weighted_loss += float(loss.detach().cpu().item()) * (end - start)
			if not uses_closure:
				optimizer.step()
			network.apply_constraints()
			optimizer.zero_grad(set_to_none=True)

		return weighted_loss / n_samples

	def fit(
		self,
		X: FloatArray,
		y: FloatArray,
		*,
		collect_history: bool = False,
		reset_components: bool = True,
		run_name: str | None = None,
		callback_updates: CallbackUpdates | None = None,
	) -> TrainingResult:
		network = self._require_model()
		self._validate_training_config()

		if not isinstance(collect_history, bool):
			raise TypeError("collect_history must be a bool.")

		if not isinstance(reset_components, bool):
			raise TypeError("reset_components must be a bool.")

		if not reset_components and (
			run_name is not None or callback_updates is not None
		):
			raise ValueError(
				"run_name and callback_updates require reset_components=True."
			)

		x_array, y_array = validate_training_arrays(
			X,
			y,
			input_spec=self.input_spec,
		)

		# Dane są lokalne dla tego wywołania, nie stanowią pól adaptera.
		x_train = torch.tensor(
			x_array, dtype=torch.float32, device=self.device
		)
		y_train = torch.tensor(
			y_array, dtype=torch.float32, device=self.device
		)

		batch_size = (
			int(x_train.shape[0])
			if self.batch_size is None
			else int(self.batch_size)
		)

		self._fitted = False
		self._loss_history = []

		state: dict[str, Any] = {
			"model": self,
			"optimizer": self.optimizer,
			"stop_training": False,
			"stop_reason": None,
			"loss_history": self._loss_history,
		}

		try:
			if reset_components:
				self.reset_components(
					run_name=run_name,
					callback_updates=callback_updates,
				)

			network.train()
			self._run_callback_hook("on_train_beginning", state)

			for epoch in range(int(self.epochs)):
				if state["stop_training"]:
					break

				state["epoch"] = epoch
				self._run_callback_hook("on_epoch_beginning", state)

				if state["stop_training"]:
					break

				epoch_loss = self._train_epoch(
					x_train,
					y_train,
					batch_size,
				)

				self._loss_history.append(epoch_loss)
				state["epoch_loss"] = epoch_loss
				state["loss"] = epoch_loss

				self._run_callback_hook("on_epoch_end", state)

				if self.verbose:
					print(
						f"Epoch {epoch + 1}/{self.epochs}"
						f" - loss: {epoch_loss:.8f}"
					)

			self._run_callback_hook("on_train_end", state)

		except Exception as exc:
			self._fitted = False
			raise TrainingError(
				f"PyTorch model training failed: {exc}"
			) from exc

		finally:
			network.eval()
			self.optimizer.zero_grad(set_to_none=True)
		
		# on_train_end may have restored weights through set_weights(),
		# which invalidates fitted status.
		self._fitted = True

		if not collect_history:
			return TrainingResult()

		iterations = len(self._loss_history)

		return TrainingResult(
			history={
				"loss": [
					float(value)
					for value in self._loss_history
				],
			},
			metrics=(
				{
					"loss": float(self._loss_history[-1]),
				}
				if self._loss_history
				else {}
			),
			n_iterations=iterations,
			stop_reason=state.get("stop_reason") or (
				"stopped_before_max_epochs"
				if iterations < int(self.epochs)
				else "max_epochs_reached"
			),
		)

	def predict(self, X: FloatArray) -> FloatArray:
		network = self._require_model()
		if not self._fitted:
			raise ModelNotFittedError("Model is not fitted. Call fit() first.")
		x = self._data_array(X, "X")
		if x.shape[1] != self._n_features:
			raise ValueError(f"Expected {self._n_features} features, got {x.shape[1]}.")
		if x.shape[0] == 0:
			return np.empty((0, int(self._n_outputs)), dtype=np.float32)
		# This architecture has no train/eval-dependent forward layers.
		# Avoid changing model/training state just to make a prediction.
		with torch.no_grad():
			result = network(torch.tensor(x, dtype=torch.float32, device=self.device))
		return np.array(result.detach().cpu().numpy(), dtype=np.float32, copy=True)

	def get_weights(self) -> ModelWeights:
		return self._require_model().get_predictor_weights()

	def set_weights(self, weights: ModelWeights) -> None:
		self._require_model().set_predictor_weights(weights)
		self._fitted = False

	def omit_variables(self, variable_indices: Sequence[int]) -> None:
		network = self._require_model()
		indices = list(variable_indices)
		if any(
			isinstance(index, (bool, np.bool_))
			or not isinstance(index, (int, np.integer))
			for index in indices
		):
			raise TypeError("Feature indices must be integers, not booleans.")
		if any(index < 0 or index >= int(self._n_features) for index in indices):
			raise ValueError("Feature index is out of range.")
		mask = np.ones(int(self._n_features), dtype=np.float32)
		if indices:
			mask[np.asarray(indices, dtype=np.intp)] = 0.0
		network.variable_control_layer.set_mask(mask)


# Backwards-compatible name; does not register a second model class.
PyTorchGrangerModel = PyTorchConditionalAdapter
