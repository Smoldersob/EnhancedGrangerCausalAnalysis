from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base_callback import Callback
from ...core.exceptions import TrainingConfigurationError


class ReduceLearningRate(Callback):
	"""Reduce optimizer learning rate after patience epochs without improvement."""

	def __init__(
		self,
		patience: int = 10,
		factor: float = 0.5,
		min_lr: float = 1e-8,
		min_delta: float = 0.0,
	) -> None:
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

		self._best_loss: float = float("inf")
		self._num_bad_epochs: int = 0

	def on_train_beginning(self, state: Dict[str, Any]) -> None:
		self._best_loss = float("inf")
		self._num_bad_epochs = 0

	def on_epoch_end(self, state: Dict[str, Any]) -> bool:
		loss = float(state.get("epoch_loss", float("nan")))
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

