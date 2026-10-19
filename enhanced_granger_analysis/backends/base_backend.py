"""Shared backend strategy interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .models.base_model import BaseGrangerModel
from ..core.components.constraint_spec import PreparedConstraintData
from ..core.components.regularizer_spec import PreparedRegularizerData

class BackendStrategy(ABC):
	"""Abstract interface for backend-specific Granger model orchestration."""
	_object_loader:Any=None
	def __init__(self, loading_verbose: bool = False) -> None:
		self._loading_verbose = bool(loading_verbose)

	def set_loading_verbose(self, value: bool) -> None:
		"""Set internal loading verbosity for component resolution logging."""
		self._loading_verbose = bool(value)

	def _consume_loading_verbose(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Pop internal loading verbosity from model config and return remaining config."""
		cfg = dict(config)
		if "loading_verbose" in cfg:
			self.set_loading_verbose(bool(cfg.pop("loading_verbose")))
		return cfg

	def _log_loaded_component(self, label: str, value: Any) -> None:
		if self._loading_verbose:
			print(f"[BackendLoader] {self.__class__.__name__}: {label} -> {value!r}")

	@abstractmethod
	def is_available(self) -> bool:
		pass

	@abstractmethod
	def build_model(
		self,
		n_features: int,
		n_outputs: int,
		regularizer: Optional[Any] = None,
		constraint: Optional[Any] = None,
		**config,
	) -> BaseGrangerModel:
		pass


	@abstractmethod
	def build_regularizer(
		self,
		regularizer_type: str | type | None,
		*,
		data: Optional[PreparedRegularizerData] = None,
	) -> Any:
		"""Construct a regularizer from a selector and prepared data.

		None means that regularization is disabled.
		User parameters are supplied only through data.params.
		"""
		raise NotImplementedError


	@abstractmethod
	def build_constraint(
		self,
		constraint_spec: Any,
		*,
		data: PreparedConstraintData,
	) -> Any:
		"""Resolve a backend constraint class and construct an instance."""
		raise NotImplementedError


	def resolve_callbacks(self, callbacks: Optional[list]) -> Optional[list]:
		"""Resolve callback specs to backend-native callback objects."""
		return callbacks

	def resolve_optimizer(self, optimizer: Any) -> Any:
		"""Resolve optimizer spec to backend-native optimizer object/class."""
		return optimizer

	