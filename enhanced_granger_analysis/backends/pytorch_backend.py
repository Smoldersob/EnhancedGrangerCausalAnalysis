from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy
from .object_loaders.torch_object_loader import TorchObjectLoader


class PyTorchBackendStrategy(BackendStrategy):
	"""Strategy for PyTorch backend."""

	@staticmethod
	def _extract_gradient_accumulation_steps(config: Dict[str, Any], optimizer_spec: Any) -> int:
		"""Extract training-level gradient accumulation setting from config/optimizer spec."""
		value = config.get("gradient_accumulation_steps", None)
		if value is None and isinstance(optimizer_spec, dict):
			optimizer_params = optimizer_spec.get("params")
			if isinstance(optimizer_params, dict):
				value = optimizer_params.get("gradient_accumulation_steps", None)
				if value is None:
					value = optimizer_params.get("gradient_accumulation", None)
				if value is None:
					value = optimizer_params.get("accumulation_steps", None)

			if value is None:
				value = optimizer_spec.get("gradient_accumulation_steps", None)
			if value is None:
				value = optimizer_spec.get("gradient_accumulation", None)
			if value is None:
				value = optimizer_spec.get("accumulation_steps", None)

		if value is None:
			return 1
		return int(value)

	@staticmethod
	def _resolve_optimizer_spec(config: Dict[str, Any]) -> Any:
		"""Normalize optimizer configuration while preserving backward compatibility.

		Supported patterns:
		- "adam"
		- {"type": "adam", "learning_rate": 1e-3}
		- {"type": "adam", "params": {"lr": 1e-3}}
		- legacy top-level "learning_rate" alongside optimizer config
		"""
		optimizer_cfg = config.get("optimizer")
		learning_rate = config.get("learning_rate")
		if optimizer_cfg is None:
			if learning_rate is None:
				return "adam"
			return {"type": "adam", "learning_rate": learning_rate}

		if isinstance(optimizer_cfg, dict):
			merged = dict(optimizer_cfg)
			params = merged.get("params")
			if learning_rate is not None:
				if isinstance(params, dict):
					params = dict(params)
					params.setdefault("lr", learning_rate)
					merged["params"] = params
				else:
					merged.setdefault("learning_rate", learning_rate)
			return merged

		if isinstance(optimizer_cfg, str) and learning_rate is not None:
			name = optimizer_cfg.strip().lower()
			if name:
				return {"type": name, "learning_rate": learning_rate}

		return optimizer_cfg

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

		model_cfg = config.get("model_config") or config.get("config") or {}
		if not isinstance(model_cfg, dict):
			model_cfg = {}
		model_name = config.get("model_name") or config.get("model") or config.get("model_type")
		model_cls = self._object_loader.resolve_model(model_name, PyTorchGrangerModel)

		regularizer_resolved = self.build_regularizer(regularizer)
		constraint_resolved = self.build_constraint(constraint)
		callbacks_resolved = self.resolve_callbacks(config.get("callbacks", None))
		optimizer_spec = self._resolve_optimizer_spec(config)
		gradient_accumulation_steps = self._extract_gradient_accumulation_steps(config, optimizer_spec)
		optimizer_resolved = self.resolve_optimizer(optimizer_spec)

		kwargs = dict(model_cfg)
		kwargs.setdefault("backend", "pytorch")
		kwargs.setdefault("regularizer", regularizer_resolved)
		kwargs.setdefault("constraint", constraint_resolved)
		kwargs.setdefault("optimizer", optimizer_resolved)
		kwargs.setdefault("loss", config.get("loss", None))
		kwargs.setdefault("callbacks", callbacks_resolved)
		kwargs.setdefault("learning_rate", config.get("learning_rate", 0.001))
		kwargs.setdefault("gradient_accumulation_steps", gradient_accumulation_steps)
		kwargs.setdefault("epochs", config.get("epochs", 100))
		kwargs.setdefault("batch_size", config.get("batch_size", 32))
		kwargs.setdefault("verbose", config.get("verbose", 0))
		kwargs.setdefault("device", config.get("device", None))
		
		return model_cls(**kwargs)

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
