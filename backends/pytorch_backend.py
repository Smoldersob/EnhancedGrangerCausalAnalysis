from __future__ import annotations

from typing import Any, Dict, List, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy
from .torch_object_loader import TorchObjectLoader


class PyTorchBackendStrategy(BackendStrategy):
	"""Strategy for PyTorch backend."""

	def __init__(self) -> None:
		self._torch = None
		self._object_loader: Optional[TorchObjectLoader] = None
		if self.is_available():
			import torch
			self._torch = torch
			self._object_loader = TorchObjectLoader(torch)

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
		scaler: Optional[Any] = None,
		**config,
	):
		from ..components.models.pytorch_model import PyTorchGrangerModel
		callbacks_resolved = self.resolve_callbacks(config.get("callbacks", None))
		optimizer_resolved = self.resolve_optimizer(config.get("optimizer", "adam"))

		return PyTorchGrangerModel(
			backend="pytorch",
			scaler=scaler,
			regularizer=regularizer,
			constraint=constraint,
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
		return self._object_loader.resolve_callbacks(callbacks)

	def resolve_optimizer(self, optimizer: Any) -> Any:
		if self._object_loader is None:
			return optimizer
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

		from ..components.constaints import build_pytorch_constraint_from_relations

		return build_pytorch_constraint_from_relations(
			relations=relations,
			predictor_names=predictor_names,
			output_names=output_names,
			col_offsets=col_offsets,
			n_features=n_features,
			base_mask=base_mask,
		)

	def build_regularizer(self, regularizer_spec: Any):
		if regularizer_spec is None:
			return None

		if isinstance(regularizer_spec, dict):
			reg_type = regularizer_spec.get("type", "l1").lower()
			if reg_type == "l1":
				from ..components.regularizers.pytorch_regularizers import PyTorchL1Regularizer
				return PyTorchL1Regularizer(l1=regularizer_spec.get("l1", 0.01))
			if reg_type == "lag_dependent_l1":
				from ..components.regularizers.pytorch_regularizers import PyTorchLagDependentL1Regularizer
				return PyTorchLagDependentL1Regularizer(
					l1=regularizer_spec.get("l1", 0.01),
					lag_weights=regularizer_spec.get("lag_weights", None),
					max_lags_per_pred=regularizer_spec.get("max_lags_per_pred", None),
					col_offsets=regularizer_spec.get("col_offsets", None),
				)

		return regularizer_spec

	def get_scaler(self):
		return None

	def get_model_hyperparameters(self) -> Dict[str, Any]:
		return {
			"epochs": 100,
			"batch_size": 32,
			"learning_rate": 0.001,
			"optimizer": "adam",
			"verbose": 0,
			"device": None,
		}

