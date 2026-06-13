from __future__ import annotations

from typing import Any, Dict, List, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy
from .object_loaders.torch_object_loader import TorchObjectLoader


class PyTorchBackendStrategy(BackendStrategy):
	"""Strategy for PyTorch backend."""

	def __init__(self, loading_verbose: bool = False) -> None:
		super().__init__(loading_verbose=loading_verbose)
		self._torch = None
		self._object_loader: Optional[TorchObjectLoader] = None
		if self.is_available():
			import torch
			self._torch = torch
			self._object_loader = TorchObjectLoader(torch, loading_verbose=loading_verbose)

	def is_available(self) -> bool:
		try:
			import torch  # noqa: F401
			return True
		except ImportError:
			return False

	def build_model(
		self,
		n_features: int,
		n_outputs: int,
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		seed: Optional[int] = None,
		**config,
	):
		if seed is not None:
			import numpy as np
			import random
			np.random.seed(seed)
			random.seed(seed)

			try:
				import torch
				torch.manual_seed(seed)
				if torch.cuda.is_available():
					torch.cuda.manual_seed_all(seed)
			except ImportError:
				pass
		from .models.pytorch_model import PyTorchGrangerModel
		config = self._consume_loading_verbose(config)
		if self._object_loader is not None:
			self._object_loader.set_loading_verbose(self._loading_verbose)

		regularizer_resolved = self.build_regularizer(regularizer)
		constraint_resolved = self.build_constraint(constraint)
		callbacks_resolved = self.resolve_callbacks(config.get("callbacks", None))
		optimizer_resolved = self.resolve_optimizer(config.get("optimizer", "adam"))
		self.validate_components(
			regularizer=regularizer_resolved,
			constraint=constraint_resolved,
			callbacks=callbacks_resolved,
			optimizer=optimizer_resolved,
		)

		return PyTorchGrangerModel(
			backend="pytorch",
			regularizer=regularizer_resolved,
			constraint=constraint_resolved,
			optimizer=optimizer_resolved,
			loss=config.get("loss", None),
			callbacks=callbacks_resolved,
			learning_rate=config.get("learning_rate", 0.001),
			epochs=config.get("epochs", 100),
			batch_size=config.get("batch_size", 32),
			verbose=config.get("verbose", 0),
			device=config.get("device", None),
		)

	def resolve_callbacks(self, callbacks: Optional[List[Any]]) -> Optional[List[Any]]:
		if self._object_loader is None:
			return callbacks
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_callbacks(callbacks)

	def resolve_optimizer(self, optimizer: Any) -> Any:
		if self._object_loader is None:
			return optimizer
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_optimizer(optimizer)

	def build_constraint_from_relations(
		self,
		relations: Dict[tuple, Any],
		predictor_names: List[str],
		output_names: List[str],
		col_offsets: NDArray,
		n_features: int,
		base_mask=None,
	):
		if not relations:
			return None

		from .constraints import build_pytorch_constraint_from_relations

		return build_pytorch_constraint_from_relations(
			relations=relations,
			predictor_names=predictor_names,
			output_names=output_names,
			col_offsets=col_offsets,
			n_features=n_features,
			base_mask=base_mask,
		)

	def build_regularizer(self, regularizer_spec: Any):
		if self._object_loader is None:
			return regularizer_spec
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_regularizer(regularizer_spec)

	def build_constraint(self, constraint_spec: Any) -> Any:
		if self._object_loader is None:
			return constraint_spec
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_constraint(constraint_spec)

	def validate_components(
		self,
		regularizer: Optional[Any],
		constraint: Optional[Any],
		callbacks: Optional[List[Any]] = None,
		optimizer: Any = None,
	) -> None:
		if self._object_loader is None:
			return
		self._object_loader.set_loading_verbose(self._loading_verbose)
		resolved_callbacks = self._object_loader.resolve_callbacks(callbacks)
		_ = self._object_loader.resolve_optimizer(optimizer)
		_ = self._object_loader.resolve_regularizer(regularizer)
		_ = self._object_loader.resolve_constraint(constraint)
		self._log_loaded_component("callbacks", resolved_callbacks)

