from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..backends import BackendFactory
from ..core.exceptions import DataValidationError
from ..core.lag_config import LagConfiguration
from ..core.outputs import MultitaskGrangerOutput
from ..preprocessing.lag.lag_engine import LagEngine
from ..preprocessing.stationarity import StationarityTransformer
from ..preprocessing.scaling import IdentityScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from ..results.granger_results import GrangerAnalysisResults


def _to_dataframe_list(data: pd.DataFrame | Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
	if isinstance(data, pd.DataFrame):
		return [data.copy()]
	if isinstance(data, Sequence) and all(isinstance(df, pd.DataFrame) for df in data):
		return [df.copy() for df in data]
	raise DataValidationError("data must be a DataFrame or a sequence of DataFrames")


def _reduce_backend_load(
	X_scaled: np.ndarray,
	y_scaled: np.ndarray,
	X_raw: np.ndarray,
	y_raw: np.ndarray,
	backend_sample_fraction: float = 1.0,
	backend_max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Reduce training workload after scaling and before backend call."""
	if not 0.0 < backend_sample_fraction <= 1.0:
		raise ValueError("backend_sample_fraction must be in range (0.0, 1.0]")

	n = X_scaled.shape[0]
	if n == 0:
		raise DataValidationError("No samples available after lag preparation")

	n_keep = max(1, int(np.floor(backend_sample_fraction * n)))
	if backend_max_samples is not None:
		if backend_max_samples <= 0:
			raise ValueError("backend_max_samples must be > 0")
		n_keep = min(n_keep, int(backend_max_samples))

	if n_keep >= n:
		return X_scaled, y_scaled, X_raw, y_raw

	idx = np.arange(n_keep, dtype=int)
	return X_scaled[idx], y_scaled[idx], X_raw[idx], y_raw[idx]


def _create_scaler(name: Optional[str]) -> Any:
	if name is None:
		return IdentityScaler()

	name_lower = name.lower()
	if name_lower in {"identity", "none"}:
		return IdentityScaler()
	if name_lower in {"standard", "zscore"}:
		return StandardScaler()
	if name_lower in {"minmax"}:
		return MinMaxScaler()
	if name_lower in {"robust"}:
		return RobustScaler()
	if name_lower in {"maxabs"}:
		return MaxAbsScaler()

	raise ValueError(f"Unsupported scaler: {name}")


class _PredictionProxy:
	"""Model-like object used to pass inverse-scaled predictions to result aggregator."""

	def __init__(self, predictions: np.ndarray, weights: List[np.ndarray]) -> None:
		self._predictions = np.asarray(predictions, dtype=np.float64)
		self._weights = [np.asarray(w, dtype=np.float64) for w in weights]

	def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002 - interface compatibility
		return self._predictions

	def get_weights(self) -> List[np.ndarray]:
		return self._weights


class MultiTaskGrangerAPI:
	"""
	Multitask orchestrator for backend-based Granger analysis.

	Pipeline:
	1) stationarity transform
	2) lag preparation (effects selection influences LagEngine target rows)
	3) scaling (X and y)
	4) optional load reduction after scaling, before backend (not validation split)
	5) base model fit
	6) reference models with hot-start + omit_variables for each tested cause
	7) inverse-scale predictions and aggregate statistics in GrangerAnalysisResults
	"""

	def __init__(self, backend: Optional[str] = None) -> None:
		self.backend = backend

	def fit(
		self,
		data: pd.DataFrame | Sequence[pd.DataFrame],
		causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		tested_causes: Optional[Sequence[str]] = None,
		relations: Optional[Mapping[Tuple[str, str], Any]] = None,
		lag_config: Optional[LagConfiguration] = None,
		lag_selector: Optional[Any] = None,
		stationarity_transformer: Optional[StationarityTransformer] = None,
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
		x_scaler: Optional[str] = "standard",
		y_scaler: Optional[str] = "standard",
		regularizer: Optional[Any] = None,
		regularizer_spec: Optional[Dict[str, Any]] = None,
		initializer: Optional[Any] = None,
		model_config: Optional[Dict[str, Any]] = None,
	) -> MultitaskGrangerOutput:
		data_list = _to_dataframe_list(data)
		all_columns = list(data_list[0].columns)

		causes_list = list(causes) if causes is not None else all_columns
		effects_list = list(effects) if effects is not None else all_columns
		tested_causes_list = list(tested_causes) if tested_causes is not None else causes_list

		unknown_causes = [c for c in causes_list if c not in all_columns]
		unknown_effects = [e for e in effects_list if e not in all_columns]
		unknown_tested = [c for c in tested_causes_list if c not in all_columns]
		if unknown_causes or unknown_effects or unknown_tested:
			raise DataValidationError(
				f"Unknown variables detected. causes={unknown_causes}, effects={unknown_effects}, tested_causes={unknown_tested}"
			)

		relations_map = dict(relations or {})

		stationarity = stationarity_transformer or StationarityTransformer()
		data_stationary = stationarity.fit_transform(data_list)

		engine = LagEngine(config=lag_config or LagConfiguration(), selector=lag_selector)
		X_all, y_all, col_offsets = engine.prepare(data_stationary, effects=effects_list)

		x_scaler_obj = _create_scaler(x_scaler)
		y_scaler_obj = _create_scaler(y_scaler)

		X_scaled = x_scaler_obj.fit_transform(X_all)
		y_scaled = y_scaler_obj.fit_transform(y_all)

		X_backend_scaled, y_backend_scaled, X_backend_raw, y_backend_raw = _reduce_backend_load(
			X_scaled=X_scaled,
			y_scaled=y_scaled,
			X_raw=X_all,
			y_raw=y_all,
			backend_sample_fraction=backend_sample_fraction,
			backend_max_samples=backend_max_samples,
		)

		strategy = BackendFactory.get_strategy(self.backend)

		reg_obj = regularizer
		if reg_obj is None and regularizer_spec is not None:
			reg_spec_mut = dict(regularizer_spec)
			if str(reg_spec_mut.get("type", "")).lower() == "lag_dependent_l1":
				max_lags = list((col_offsets[1:] - col_offsets[:-1]).astype(int))
				reg_spec_mut.setdefault("max_lags_per_pred", max_lags)
				reg_spec_mut.setdefault("col_offsets", list(col_offsets[:-1].astype(int)))
			reg_obj = strategy.build_regularizer(reg_spec_mut)

		base_mask = None
		if engine.mask_ is not None:
			mask_arr = np.asarray(engine.mask_, dtype=np.float64)
			if mask_arr.shape[0] == len(effects_list):
				base_mask = mask_arr
			else:
				effect_indices = [all_columns.index(e) for e in effects_list]
				base_mask = mask_arr[effect_indices, :]

		constraint_obj = strategy.build_constraint_from_relations(
			relations=relations_map,
			predictor_names=all_columns,
			output_names=effects_list,
			col_offsets=col_offsets[:-1],
			n_features=X_all.shape[1],
			base_mask=base_mask,
		)

		model_cfg = dict(model_config or {})
		base_model = strategy.build_model(
			n_features=X_all.shape[1],
			n_outputs=y_all.shape[1],
			regularizer=reg_obj,
			constraint=constraint_obj,
			scaler=None,
			**model_cfg,
		)
		base_model.initialize(X_backend_scaled, targets=y_backend_scaled)

		if initializer is not None:
			self._apply_initializer(
				model=base_model,
				initializer=initializer,
				X_train=X_backend_scaled,
				y_train=y_backend_scaled,
				mask=base_mask,
			)

		base_model.fit()
		base_weights = base_model.get_weights()

		base_pred_scaled = np.asarray(base_model.predict(X_backend_scaled), dtype=np.float64)
		base_pred_real = y_scaler_obj.inverse_transform(base_pred_scaled)
		y_backend_real = y_scaler_obj.inverse_transform(y_backend_scaled)

		results = GrangerAnalysisResults(effects=effects_list, causes=tested_causes_list)
		reference_models: Dict[str, Any] = {}

		for cause_name in tested_causes_list:
			cause_idx = all_columns.index(cause_name)

			ref_model = strategy.build_model(
				n_features=X_all.shape[1],
				n_outputs=y_all.shape[1],
				regularizer=reg_obj,
				constraint=constraint_obj,
				scaler=None,
				**model_cfg,
			)
			ref_model.initialize(X_backend_scaled, targets=y_backend_scaled)
			ref_model.set_weights(base_weights)

			start = int(col_offsets[cause_idx])
			end = int(col_offsets[cause_idx + 1])
			ref_model.omit_variables(list(range(start, end)))
			ref_model.fit()

			ref_pred_scaled = np.asarray(ref_model.predict(X_backend_scaled), dtype=np.float64)
			ref_pred_real = y_scaler_obj.inverse_transform(ref_pred_scaled)

			base_proxy = _PredictionProxy(predictions=base_pred_real, weights=base_weights)
			ref_proxy = _PredictionProxy(predictions=ref_pred_real, weights=ref_model.get_weights())

			results.update_cause(
				cause=cause_name,
				cause_index=cause_idx,
				base_model=base_proxy,
				reference_model=ref_proxy,
				X=X_backend_raw,
				y=y_backend_real,
				col_offsets=col_offsets,
			)
			reference_models[cause_name] = ref_model

		return MultitaskGrangerOutput(
			results=results,
			base_model=base_model,
			reference_models=reference_models,
			stationarity_transformer=stationarity,
			lag_engine=engine,
			X_scaler=x_scaler_obj,
			y_scaler=y_scaler_obj,
		)

	@staticmethod
	def _apply_initializer(
		model: Any,
		initializer: Any,
		X_train: np.ndarray,
		y_train: np.ndarray,
		mask: Optional[np.ndarray],
	) -> None:
		"""Initialize base model weights before first fit."""
		if isinstance(initializer, type):
			init_obj = initializer(
				n_targets=y_train.shape[1],
				n_features_eff=X_train.shape[1],
			)
		else:
			init_obj = initializer

		if not callable(init_obj):
			raise TypeError("initializer must be callable or an initializer class")

		try:
			A, B = init_obj(Y=y_train, X_lagged=X_train, mask=mask)
		except TypeError:
			A, B = init_obj()

		A_arr = np.asarray(A, dtype=np.float64)
		B_arr = np.asarray(B, dtype=np.float64)

		# Initializers produce A as (n_outputs, n_features); models accept kernel as (n_features, n_outputs).
		kernel = A_arr.T
		model.set_weights([kernel, B_arr])


__all__ = ["MultiTaskGrangerAPI", "MultitaskGrangerOutput"]
