from __future__ import annotations

from typing import Any, Dict, List, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy


class TensorFlowBackendStrategy(BackendStrategy):
	"""Strategy for TensorFlow/Keras backend."""

	def __init__(self) -> None:
		self._tf = None
		self._keras = None
		if self.is_available():
			import tensorflow as tf
			self._tf = tf
			self._keras = tf.keras

	def is_available(self) -> bool:
		try:
			import tensorflow  # noqa: F401
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
		from ..components.models.tensorflow_model import TensorFlowGrangerModel

		return TensorFlowGrangerModel(
			backend="tensorflow",
			scaler=scaler,
			regularizer=regularizer,
			constraint=constraint,
			optimizer=config.get("optimizer", "adam"),
			loss=config.get("loss", "mse"),
			callbacks=config.get("callbacks", None),
			epochs=config.get("epochs", 100),
			batch_size=config.get("batch_size", 32),
			verbose=config.get("verbose", 0),
		)

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

		from ..components.constaints import build_tensorflow_constraint_from_relations

		return build_tensorflow_constraint_from_relations(
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
				from ..components.regularizers.tensorflow_regularizers import KerasL1Regularizer
				return KerasL1Regularizer(l1=regularizer_spec.get("l1", 0.01))
			if reg_type == "lag_dependent_l1":
				from ..components.regularizers.tensorflow_regularizers import KerasLagDependentL1Regularizer
				return KerasLagDependentL1Regularizer(
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
			"loss": "mse",
			"verbose": 0,
		}

