from __future__ import annotations

from abc import ABC
from typing import Any, Dict


class Callback(ABC):
	"""Base callback interface for PyTorch training loop hooks."""

	def on_train_beginning(self, state: Dict[str, Any]) -> None:
		"""Called once before training starts."""

	def on_epoch_beginning(self, state: Dict[str, Any]) -> bool:
		"""Called at the beginning of each epoch. Return False to stop training."""
		return True

	def on_epoch_end(self, state: Dict[str, Any]) -> bool:
		"""Called at the end of each epoch. Return False to stop training."""
		return True

	def on_train_end(self, state: Dict[str, Any]) -> None:
		"""Called once after training ends."""

