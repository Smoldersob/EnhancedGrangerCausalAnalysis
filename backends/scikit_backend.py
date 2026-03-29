from __future__ import annotations

from typing import Any, Dict, List, Optional

from numpy.typing import NDArray

from .base_backend import BackendStrategy


class ScikitBackendStrategy(BackendStrategy):
	"""Strategy for scikit-learn backend."""

	def __init__(self) -> None:
		self._sklearn = None
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
		scaler: Optional[Any] = None,
		**config,
	):
		from ..components.models.scikit_model import ScikitConstrainedGrangerModel

		return ScikitConstrainedGrangerModel(
			backend="sklearn",
			scaler=scaler,
			regularizer=regularizer,
			constraint=constraint,
			callbacks=config.get("callbacks", None),
			fit_intercept=config.get("fit_intercept", True),
			learning_rate=config.get("learning_rate", 1.0),
			max_iter=config.get("max_iter", 1000),
			tol=config.get("tol", 1e-8),
			batch_size=config.get("batch_size", None),
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

		from ..components.constaints import build_numpy_constraint_from_relations

		return build_numpy_constraint_from_relations(
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
				from ..components.regularizers.numpy_regularizers import NumpyL1Regularizer
				return NumpyL1Regularizer(l1=regularizer_spec.get("l1", 0.01))
			if reg_type == "lag_dependent_l1":
				from ..components.regularizers.numpy_regularizers import NumpyLagDependentL1Regularizer
				return NumpyLagDependentL1Regularizer(
					l1=regularizer_spec.get("l1", 0.01),
					lag_weights=regularizer_spec.get("lag_weights", None),
					max_lags_per_pred=regularizer_spec.get("max_lags_per_pred", None),
					col_offsets=regularizer_spec.get("col_offsets", None),
				)

		return regularizer_spec

	def get_scaler(self):
		try:
			from sklearn.preprocessing import StandardScaler
			return StandardScaler()
		except ImportError:
			return None

	def get_model_hyperparameters(self) -> Dict[str, Any]:
		return {
			"max_iter": 1000,
			"learning_rate": 1.0,
			"tol": 1e-8,
			"batch_size": None,
			"fit_intercept": True,
			"verbose": 0,
		}

