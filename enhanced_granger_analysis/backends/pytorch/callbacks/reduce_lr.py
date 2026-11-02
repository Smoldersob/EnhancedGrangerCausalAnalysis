from __future__ import annotations

from typing import Any, Dict, Optional, Mapping

import numpy as np

from .base import Callback
from .registry import register_callback
from ....core.exceptions import TrainingConfigurationError

@register_callback(
    "reduce_lr",
    "reduce_learning_rate",
    "reducelearningrate",
    "reduce_lr_on_plateau",
)
class ReduceLearningRate(Callback):
	"""Reduce optimizer learning rate after patience epochs without improvement."""

	def __init__(
		self,
		patience: int = 10,
		factor: float = 0.5,
		min_lr: float = 1e-8,
		min_delta: float = 0.0,
		monitor="epoch_loss",
	) :
		if patience <= 0:
			raise TrainingConfigurationError("patience must be a positive integer")
		if not (0.0 < factor < 1.0):
			raise TrainingConfigurationError("factor must be in range (0, 1)")
		if min_lr < 0:
			raise TrainingConfigurationError("min_lr must be >= 0")
		if min_delta < 0:
			raise TrainingConfigurationError("min_delta must be >= 0")

		self.patience = patience
		self.factor = factor
		self.min_lr = min_lr
		self.min_delta = min_delta
		self.monitor = monitor

		self._best_loss: float = float("inf")
		self._num_bad_epochs: int = 0

	def on_train_beginning(self, state: Dict[str, Any]) -> None:
		self._best_loss = float("inf")
		self._num_bad_epochs = 0

	def on_epoch_end(self, state: Dict[str, Any]) -> bool:
		loss = float(state.get(self.monitor, float("nan")))
		optimizer = state.get("optimizer")

		if optimizer is None or np.isnan(loss):
			return True

		improved = (self._best_loss - loss) > self.min_delta
		if improved:
			self._best_loss = loss
			self._num_bad_epochs = 0
			return True

		self._num_bad_epochs += 1
		if self._num_bad_epochs < self.patience:
			return True

		for param_group in optimizer.param_groups:
			current_lr = float(param_group.get("lr", 0.0))
			new_lr = max(current_lr * self.factor, self.min_lr)
			param_group["lr"] = new_lr

		self._num_bad_epochs = 0
		return True

	def clone_for_run(self, run_name: str, callback_updates: Optional[Mapping[str, Any]] = None) -> "ReduceLearningRate":
		"""Create a clean callback; do not copy the best loss or stored weights."""
		del run_name  # This callback has no run-specific destination.

		if callback_updates:
			patience = callback_updates.get("patience", self.patience)
			factor = callback_updates.get("factor", self.factor)
			min_lr = callback_updates.get("min_lr", self.min_lr)
			min_delta = callback_updates.get("min_delta", self.min_delta)
			monitor = callback_updates.get("monitor", self.monitor)
		else:
			patience = self.patience
			factor = self.factor
			min_lr = self.min_lr
			min_delta = self.min_delta
			monitor = self.monitor

		return type(self)(
			patience=patience,
			factor=factor,
			min_lr=min_lr,
			min_delta=min_delta,
			monitor=monitor,
		)