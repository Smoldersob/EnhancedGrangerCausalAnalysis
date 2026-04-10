from __future__ import annotations

from typing import Any, Dict, List, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy
from .object_loaders.np_object_loader import NumpyObjectLoader


class ScikitBackendStrategy(BackendStrategy):
	"""Strategy for scikit-learn backend."""

	def __init__(self, loading_verbose: bool = False) -> None:
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
		**config,
	):
		from .models.scikit_model import ScikitConstrainedGrangerModel
		config = self._consume_loading_verbose(config)
		self._object_loader.set_loading_verbose(self._loading_verbose)

		regularizer_resolved = self.build_regularizer(regularizer)
		constraint_resolved = self.build_constraint(constraint)
		callbacks_resolved = self.resolve_callbacks(config.get("callbacks", None))
		optimizer_resolved = self.resolve_optimizer(config.get("optimizer", None))
		self.validate_components(
			regularizer=regularizer_resolved,
			constraint=constraint_resolved,
			callbacks=callbacks_resolved,
			optimizer=optimizer_resolved,
		)

		return ScikitConstrainedGrangerModel(
			backend="sklearn",
			regularizer=regularizer_resolved,
			constraint=constraint_resolved,
			callbacks=callbacks_resolved,
			fit_intercept=config.get("fit_intercept", True),
			learning_rate=config.get("learning_rate", 1.0),
			max_iter=config.get("max_iter", 1000),
			tol=config.get("tol", 1e-8),
			batch_size=config.get("batch_size", None),
			verbose=config.get("verbose", 0),
		)

	def resolve_callbacks(self, callbacks: Optional[List[Any]]) -> Optional[List[Any]]:
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_callbacks(callbacks)

	def resolve_optimizer(self, optimizer: Any) -> Any:
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

		from .constraints import build_numpy_constraint_from_relations

		return build_numpy_constraint_from_relations(
			relations=relations,
			predictor_names=predictor_names,
			output_names=output_names,
			col_offsets=col_offsets,
			n_features=n_features,
			base_mask=base_mask,
		)

	def build_regularizer(self, regularizer_spec: Any):
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_regularizer(regularizer_spec)

	def build_constraint(self, constraint_spec: Any) -> Any:
		self._object_loader.set_loading_verbose(self._loading_verbose)
		return self._object_loader.resolve_constraint(constraint_spec)

	def validate_components(
		self,
		regularizer: Optional[Any],
		constraint: Optional[Any],
		callbacks: Optional[List[Any]] = None,
		optimizer: Any = None,
	) -> None:
		self._object_loader.set_loading_verbose(self._loading_verbose)
		resolved_callbacks = self._object_loader.resolve_callbacks(callbacks)
		_ = self._object_loader.resolve_optimizer(optimizer)
		_ = self._object_loader.resolve_regularizer(regularizer)
		_ = self._object_loader.resolve_constraint(constraint)
		self._log_loaded_component("callbacks", resolved_callbacks)


