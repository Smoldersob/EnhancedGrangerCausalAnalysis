from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base_callback import Callback
from ...core.exceptions import TrainingConfigurationError


class ConvergenceCheck(Callback):
	"""Stop training when relative epoch-to-epoch loss change is below threshold."""

	def __init__(self, relative_change_threshold: float = 1e-4) -> None:
		if relative_change_threshold < 0:
			raise TrainingConfigurationError("relative_change_threshold must be >= 0")
		self.relative_change_threshold = relative_change_threshold

	def on_epoch_end(self, state: Dict[str, Any]) -> bool:
		loss_history = state.get("loss_history") or []
		if len(loss_history) < 2:
			return True

		prev_loss = float(loss_history[-2])
		curr_loss = float(loss_history[-1])
		if np.isnan(prev_loss) or np.isnan(curr_loss):
			return True

		denominator = max(abs(prev_loss), 1e-12)
		rel_change = abs(prev_loss - curr_loss) / denominator

		if rel_change < self.relative_change_threshold:
			state["stop_training"] = True
			state["stop_reason"] = "convergence_check"
			return False

		return True

