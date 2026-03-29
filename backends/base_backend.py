"""Shared backend strategy interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from ..components.models.base_model import BaseGrangerModel


class BackendStrategy(ABC):
	"""Abstract interface for backend-specific Granger model orchestration."""

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
		scaler: Optional[Any] = None,
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

	def get_scaler(self) -> Optional[Any]:
		return None

	@abstractmethod
	def get_model_hyperparameters(self) -> Dict[str, Any]:
		pass

	def validate_components(
		self,
		regularizer: Optional[Any],
		constraint: Optional[Any],
	) -> None:
		pass
