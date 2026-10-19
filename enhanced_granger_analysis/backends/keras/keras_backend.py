import importlib
if importlib.util.find_spec("keras"):
	import os
	if importlib.util.find_spec("tensorflow"):
		os.environ["KERAS_BACKEND"] = "tensorflow"
	elif importlib.util.find_spec("torch"):
		os.environ["KERAS_BACKEND"] = "torch"

from typing import Any, Dict, Optional, Mapping

from ..base_backend import BackendStrategy
from ..object_loaders.keras_object_loader import KerasObjectLoader
from ...core.components.constraint_spec import PreparedConstraintData
from ...core.exceptions import ConstraintConfigurationError
from ...core.components.regularizer_spec import PreparedRegularizerData


class KerasBackendStrategy(BackendStrategy):
	"""Strategy for Keras backend."""

	@staticmethod
	def _resolve_optimizer_spec(config: Dict[str, Any]) -> Any:
		"""Normalize optimizer configuration while preserving backward compatibility.

		Supported patterns:
		- "adam"
		- {"type": "adam", "learning_rate": 1e-3}
		- {"type": "adam", "params": {"learning_rate": 1e-3}}
		- legacy top-level "learning_rate" alongside optimizer config
		- nested model_config.* values, including model_config.learning_rate
		"""
		model_cfg = config.get("model_config") or config.get("config") or {}
		if not isinstance(model_cfg, Mapping):
			model_cfg = {}

		optimizer_cfg = config.get("optimizer")
		if optimizer_cfg is None:
			optimizer_cfg = model_cfg.get("optimizer")

		learning_rate = config.get("learning_rate")
		if learning_rate is None:
			learning_rate = model_cfg.get("learning_rate")

		if optimizer_cfg is None:
			if learning_rate is None:
				return "adam"
			return {"type": "adam", "learning_rate": learning_rate}

		if isinstance(optimizer_cfg, Mapping):
			merged = dict(optimizer_cfg)
			params = merged.get("params")
			if isinstance(params, dict):
				params = dict(params)
			else:
				params = {}
			if learning_rate is not None:
				params.setdefault("learning_rate", learning_rate)
				merged["params"] = params
			if "learning_rate" in merged and "learning_rate" not in params:
				params.setdefault("learning_rate", merged["learning_rate"])
				merged["params"] = params
			return merged

		if isinstance(optimizer_cfg, str) and learning_rate is not None:
			name = optimizer_cfg.strip().lower()
			if name:
				return {"type": name, "learning_rate": learning_rate}

		return optimizer_cfg

	def __init__(self, loading_verbose: bool = False):
		super().__init__(loading_verbose=loading_verbose)
		self._object_loader: Optional[KerasObjectLoader] = None
		if self.is_available():
			import keras
			self._object_loader = KerasObjectLoader(keras, loading_verbose=loading_verbose)

	def is_available(self) -> bool:
		return importlib.util.find_spec("keras")

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
				if importlib.util.find_spec("keras") is not None:
					import keras
					keras.utils.set_random_seed(seed)
			except ImportError:
				pass
			try:
				if importlib.util.find_spec("tensorflow") is not None:
					import tensorflow as tf
					tf.random.set_seed(seed)
			except ImportError:
				pass
			try:
				if importlib.util.find_spec("torch") is not None:
					import torch
					torch.manual_seed(seed)
					if torch.cuda.is_available():
						torch.cuda.manual_seed_all(seed)
			except ImportError:
				pass

		from ..models.keras_model import KerasGrangerModel
		config = self._consume_loading_verbose(config)
		if self._object_loader is not None:
			self._object_loader.set_loading_verbose(self._loading_verbose)

		model_cfg = config.get("model_config") or config.get("config") or {}
		if not isinstance(model_cfg, dict):
			model_cfg = {}
		model_name = config.get("model_name") or config.get("model") or config.get("model_type")
		model_cls = self._object_loader.resolve_model(model_name, KerasGrangerModel)

		callbacks_cfg = config.get("callbacks", None)
		callbacks_resolved = self.resolve_callbacks(callbacks_cfg)
		optimizer_spec = self._resolve_optimizer_spec(config)
		optimizer_resolved = self.resolve_optimizer(optimizer_spec)

		kwargs = dict(model_cfg)
		if "constraint" in kwargs or "constraint_spec" in kwargs:
			raise ConstraintConfigurationError(
				"Do not configure constraints inside model_config. "
				"Use the builder's constraint_spec setting."
			)

		kwargs["constraint"] = constraint
		kwargs["regularizer"] = regularizer
		
		kwargs.setdefault("backend", "keras")
		kwargs.setdefault("optimizer", optimizer_resolved)
		kwargs.setdefault("loss", config.get("loss", "mse"))
		kwargs.setdefault("callbacks", callbacks_resolved)
		kwargs.setdefault("epochs", config.get("epochs", 100))
		kwargs.setdefault("batch_size", config.get("batch_size", 32))
		kwargs.setdefault("verbose", config.get("verbose", 0))
		kwargs.setdefault("device", config.get("device", None))

		return model_cls(**kwargs)

	def resolve_callbacks(self, callbacks: Optional[list]) -> Optional[list]:
		if self._object_loader is None:
			return callbacks
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_callbacks(callbacks)

	def resolve_optimizer(self, optimizer: Any) -> Any:
		if self._object_loader is None:
			return optimizer
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_optimizer(optimizer)

	def build_regularizer(
		self,
		regularizer_type: str | type | None,
		*,
		data: Optional[PreparedRegularizerData] = None,
	) -> Any:
		if regularizer_type is None:
			return None

		from .regularizers import resolve_regularizer_spec

		regularizer_class, _ = resolve_regularizer_spec(
			regularizer_type
		)

		return regularizer_class(data=data)

	def build_constraint(
		self,
		constraint_spec: Any,
		*,
		data: PreparedConstraintData,
	) -> Any:
		from .constraints import resolve_constraint_spec

		constraint_class, params = resolve_constraint_spec(
			constraint_spec
		)

		constraint = constraint_class(
			data=data,
			**params,
		)

		self._log_loaded_component("constraint", constraint)
		return constraint


