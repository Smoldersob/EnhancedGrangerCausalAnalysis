from __future__ import annotations

from typing import Any, Optional

from ..base_backend import BackendStrategy
from ..object_loaders.np_object_loader import NumpyObjectLoader
from ...core.components.constraint_spec import PreparedConstraintData
from ...core.exceptions import ConstraintConfigurationError
from ...core.components.regularizer_spec import PreparedRegularizerData

class ScikitBackendStrategy(BackendStrategy):
	"""Strategy for scikit-learn backend."""

	def __init__(self, loading_verbose: bool = False):
		super().__init__(loading_verbose=loading_verbose)
		self._sklearn = None
		self._object_loader = NumpyObjectLoader(loading_verbose=loading_verbose)
		if self.is_available():
			import sklearn  # noqa: F401
			self._sklearn = sklearn

	def is_available(self) -> bool:
		try:
			import sklearn  # noqa: F401
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

		from ..models.scikit_model import ScikitConstrainedGrangerModel
		config = self._consume_loading_verbose(config)
		self._object_loader.set_loading_verbose(self._loading_verbose)

		model_cfg = config.get("model_config") or config.get("config") or {}
		if not isinstance(model_cfg, dict):
			model_cfg = {}
		model_name = config.get("model_name") or config.get("model") or config.get("model_type")
		model_cls = self._object_loader.resolve_model(model_name, ScikitConstrainedGrangerModel)

		callbacks_resolved = self.resolve_callbacks(config.get("callbacks", None))

		kwargs = dict(model_cfg)
		if "constraint" in kwargs or "constraint_spec" in kwargs:
			raise ConstraintConfigurationError(
				"Do not configure constraints inside model_config. "
				"Use the builder's constraint_spec setting."
			)

		kwargs["constraint"] = constraint
		kwargs["regularizer"] = regularizer
				
		kwargs.setdefault("backend", "sklearn")
		kwargs.setdefault("regularizer", regularizer_resolved)
		kwargs.setdefault("callbacks", callbacks_resolved)
		kwargs.setdefault("fit_intercept", config.get("fit_intercept", True))
		kwargs.setdefault("learning_rate", config.get("learning_rate", 1.0))
		kwargs.setdefault("max_iter", config.get("max_iter", 1000))
		kwargs.setdefault("tol", config.get("tol", 1e-8))
		kwargs.setdefault("batch_size", config.get("batch_size", None))
		kwargs.setdefault("verbose", config.get("verbose", 0))

		return model_cls(**kwargs)

	def resolve_callbacks(self, callbacks: Optional[list]) -> Optional[list]:
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_callbacks(callbacks)

	def resolve_optimizer(self, optimizer: Any) -> Any:
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