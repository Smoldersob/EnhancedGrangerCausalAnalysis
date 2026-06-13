from __future__ import annotations

import importlib
import os
import re
from typing import Any, Dict, Optional

import numpy as np

from .base_callback import Callback
from ...core.exceptions import TrainingConfigurationError, TrainingError


class TorchTensorBoardCallback(Callback):
	"""TensorBoard-style logging callback for PyTorch training."""

	def __init__(
		self,
		log_dir: str = "runs/enhanced_granger_analysis",
		log_every_n_epochs: int = 1,
		flush_secs: int = 30,
		track_weight_histograms: bool = False,
		histogram_every_n_epochs: int = 1,
	) -> None:
		if log_every_n_epochs <= 0:
			raise TrainingConfigurationError("log_every_n_epochs must be a positive integer")
		if histogram_every_n_epochs <= 0:
			raise TrainingConfigurationError("histogram_every_n_epochs must be a positive integer")

		self.log_dir = log_dir
		self.log_every_n_epochs = log_every_n_epochs
		self.flush_secs = flush_secs
		self.track_weight_histograms = track_weight_histograms
		self.histogram_every_n_epochs = histogram_every_n_epochs
		self._run_name: Optional[str] = None

		self._writer: Optional[Any] = None

	@staticmethod
	def _sanitize_segment(name: str) -> str:
		return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_") or "run"

	def set_run_name(self, run_name: str) -> None:
		"""Set run name and update log_dir before writer initialization."""
		if self._writer is not None:
			raise TrainingError("Cannot change run_name after writer was initialized")
		self._run_name = run_name
		self.log_dir = os.path.join(self.log_dir, self._sanitize_segment(run_name))

	def clone_for_run(self, run_name: str) -> "TorchTensorBoardCallback":
		"""Return a fresh callback instance bound to a run-specific subdirectory."""
		clone = TorchTensorBoardCallback(
			log_dir=self.log_dir,
			log_every_n_epochs=self.log_every_n_epochs,
			flush_secs=self.flush_secs,
			track_weight_histograms=self.track_weight_histograms,
			histogram_every_n_epochs=self.histogram_every_n_epochs,
		)
		clone.set_run_name(run_name)
		return clone

	def _ensure_writer(self) -> None:
		"""Lazily initialize SummaryWriter from torch utilities."""
		if self._writer is not None:
			return

		try:
			tensorboard_module = importlib.import_module("torch.utils.tensorboard")
		except Exception as exc:  # pragma: no cover - optional runtime dependency
			raise TrainingError(
				"torch.utils.tensorboard is unavailable. Install PyTorch with tensorboard support."
			) from exc

		self._writer = tensorboard_module.SummaryWriter(
			log_dir=self.log_dir,
			flush_secs=self.flush_secs,
		)

	def on_train_beginning(self, state: Dict[str, Any]) -> None:
		self._ensure_writer()

	def on_epoch_end(self, state: Dict[str, Any]) -> bool:
		if self._writer is None:
			return True

		epoch = int(state.get("epoch", 0))
		step = epoch + 1

		if step % self.log_every_n_epochs != 0:
			return True

		loss = float(state.get("epoch_loss", float("nan")))
		if not np.isnan(loss):
			self._writer.add_scalar("train/loss", loss, step)

		optimizer = state.get("optimizer")
		if optimizer is not None and getattr(optimizer, "param_groups", None):
			lr = float(optimizer.param_groups[0].get("lr", 0.0))
			self._writer.add_scalar("train/learning_rate", lr, step)

		if self.track_weight_histograms and step % self.histogram_every_n_epochs == 0:
			model = state.get("model")
			coefficient_layer = getattr(model, "_coefficient_layer", None)
			if coefficient_layer is not None and getattr(coefficient_layer, "weight", None) is not None:
				weights = coefficient_layer.weight.detach().cpu()
				self._writer.add_histogram("weights/coefficients", weights, step)

		self._writer.flush()
		return True

	def on_train_end(self, state: Dict[str, Any]) -> None:
		if self._writer is not None:
			self._writer.flush()
			self._writer.close()
			self._writer = None
