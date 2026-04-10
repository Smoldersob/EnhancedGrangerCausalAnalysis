from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base_callback import Callback
from ...core.exceptions import TrainingConfigurationError


class EarlyStopping(Callback):
	"""Stop training after patience epochs without loss improvement."""

	def __init__(
		self,
		patience: int = 20,
		min_delta: float = 0.0,
		restore_best_weights: bool = True,
	) -> None:
		if patience <= 0:
			raise TrainingConfigurationError("patience must be a positive integer")
		if min_delta < 0:
			raise TrainingConfigurationError("min_delta must be >= 0")

		self.patience = patience
		self.min_delta = min_delta
		self.restore_best_weights = restore_best_weights

		self._best_loss: float = float("inf")
		self._num_bad_epochs: int = 0
		self._best_weights: Optional[List[np.ndarray]] = None

	def on_train_beginning(self, state: Dict[str, Any]) -> None:
		self._best_loss = float("inf")
		self._num_bad_epochs = 0
		self._best_weights = None

	def on_epoch_end(self, state: Dict[str, Any]) -> bool:
		loss = float(state.get("epoch_loss", float("nan")))
		model = state.get("model")

		if np.isnan(loss):
			return True

		improved = (self._best_loss - loss) > self.min_delta
		if improved:
			self._best_loss = loss
			self._num_bad_epochs = 0
			if self.restore_best_weights and model is not None:
				self._best_weights = model.get_weights()
		else:
			self._num_bad_epochs += 1

		if self._num_bad_epochs >= self.patience:
			state["stop_training"] = True
			state["stop_reason"] = "early_stopping"
			return False

		return True

	def on_train_end(self, state: Dict[str, Any]) -> None:
		model = state.get("model")
		if self.restore_best_weights and self._best_weights is not None and model is not None:
			model.set_weights(self._best_weights)

