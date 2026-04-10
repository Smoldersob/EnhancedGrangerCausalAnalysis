"""Shared backend strategy interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .models.base_model import BaseGrangerModel


class BackendStrategy(ABC):
	"""Abstract interface for backend-specific Granger model orchestration."""

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
	def build_constraint_from_relations(
		self,
		relations: Dict[tuple, Any],
		predictor_names: List[str],
		output_names: List[str],
		col_offsets: NDArray[np.int_],
		n_features: int,
		base_mask: Optional[NDArray[np.float64]] = None,
	) -> Any:
		pass

	@abstractmethod
	def build_regularizer(self, regularizer_spec: Any) -> Any:
		pass

	def build_constraint(self, constraint_spec: Any) -> Any:
		"""Resolve backend-native constraint object from user-provided spec/object."""
		return constraint_spec


	def validate_components(
		self,
		regularizer: Optional[Any],
		constraint: Optional[Any],
		callbacks: Optional[List[Any]] = None,
		optimizer: Any = None,
	) -> None:
		pass

	def resolve_callbacks(self, callbacks: Optional[List[Any]]) -> Optional[List[Any]]:
		"""Resolve callback specs to backend-native callback objects."""
		return callbacks

	def resolve_optimizer(self, optimizer: Any) -> Any:
		"""Resolve optimizer spec to backend-native optimizer object/class."""
		return optimizer
