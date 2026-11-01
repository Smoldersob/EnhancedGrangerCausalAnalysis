from __future__ import annotations

from ..preprocessing.prepare_data import PreparedData
from ..core.models.protocol import (
	CallbackUpdates,
	FloatArray,
	GrangerModelProtocol,
)
from ..core.models.training_result import TrainingResult


class FitExecutor:
	"""Execute a single fit through the existing model protocol.

	The caller selects the observations and prepares model weights,
	input-control state, and any explicit component resets.
	"""

	def fit(
		self,
		*,
		model: GrangerModelProtocol,
		X: FloatArray,
		y: FloatArray,
		collect_history: bool = False,
		reset_components: bool = True,
		run_name: str | None = None,
		callback_updates: CallbackUpdates | None = None,
	) -> TrainingResult:
		"""Fit a model without changing the surrounding execution policy."""

		return model.fit(
			X,
			y,
			collect_history=collect_history,
			reset_components=reset_components,
			run_name=run_name,
			callback_updates=callback_updates,
		)

	def fit_training(
		self,
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		collect_history: bool = False,
		reset_components: bool = True,
		run_name: str | None = None,
		callback_updates: CallbackUpdates | None = None,
	) -> TrainingResult:
		"""Fit on the training part of prepared observations."""
		X, y = prepared.arrays_for(
			"training",
			scaled=True,
			require_nonempty=True,
		)

		return self.fit(
			model=model,
			X=X,
			y=y,
			collect_history=collect_history,
			reset_components=reset_components,
			run_name=run_name,
			callback_updates=callback_updates,
		)