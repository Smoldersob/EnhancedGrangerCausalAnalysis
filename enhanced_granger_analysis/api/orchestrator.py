from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import time

from ..backends import BackendFactory
from ..core.lag_config import LagConfiguration
from ..preprocessing.stationarity import StationarityTransformer
from ..preprocessing.lag import BaseLagSelector
from ..preprocessing.prepare_data import PreparedData, prepare_data

from ..results.granger_results import GrangerAnalysisResults, ResultsError, PreparationContext
from ..hiperopt.hyperoptimization import MultiTaskGrangerHyperparameterOptimizer

# Preprocessing utility functions for run-specific callback handling and regularizer specification.
def _sanitize_run_name(name: str) -> str:
	"""Convert arbitrary run identifier into a safe path segment."""
	return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_") or "run"


def _clone_callback_for_run(callback: Any, run_name: str) -> Any:
	"""Create run-scoped callback instance and adjust log destination when supported."""
	if hasattr(callback, "clone_for_run") and callable(getattr(callback, "clone_for_run")):
		try:
			return callback.clone_for_run(run_name)
		except Exception:
			pass

	cloned: Any = None
	if hasattr(callback, "get_config") and hasattr(callback.__class__, "from_config"):
		try:
			cfg = callback.get_config()
			cloned = callback.__class__.from_config(cfg)
		except Exception:
			cloned = None

	if cloned is None:
		try:
			cloned = copy.deepcopy(callback)
		except Exception:
			cloned = callback

	if cloned is callback:
		# Fallback: callback is not safely clonable, keep original unchanged.
		return callback

	if hasattr(cloned, "set_run_name") and callable(getattr(cloned, "set_run_name")):
		try:
			cloned.set_run_name(run_name)
		except Exception:
			pass
	elif isinstance(getattr(cloned, "log_dir", None), str):
		cloned.log_dir = os.path.join(cloned.log_dir, _sanitize_run_name(run_name))

	return cloned


def _build_callbacks_for_run(callbacks: Optional[Sequence[Any]], run_name: str) -> Optional[List[Any]]:
	"""Return callback list cloned and specialized for a training run."""
	if callbacks is None:
		return None
	return [_clone_callback_for_run(cb, run_name) for cb in callbacks]


def _regularizer_spec_from_value(regularizer: Any) -> Optional[Dict[str, Any]]:
	if regularizer is None:
		return None

	if isinstance(regularizer, Mapping):
		return dict(regularizer)

	params: Dict[str, Any]
	if hasattr(regularizer, "get_config") and callable(getattr(regularizer, "get_config")):
		config = regularizer.get_config()
		if isinstance(config, Mapping):
			params = dict(config)
		else:
			params = {}
	elif hasattr(regularizer, "get_params") and callable(getattr(regularizer, "get_params")):
		config = regularizer.get_params()
		params = dict(config) if isinstance(config, Mapping) else {}
	else:
		params = {}

	class_name = regularizer.__class__.__name__.lower()
	if "lagdependent" in class_name or "lag_dependent" in class_name:
		type_name = "lag_dependent_l1"
	elif class_name.endswith("l1regularizer") or class_name.endswith("l1"):
		type_name = "l1"
	else:
		return None

	params.setdefault("type", type_name)
	return params

class MultiTaskGrangerAPI:
	"""
	Multitask orchestrator for backend-based Granger analysis.

	Pipeline:
	1) stationarity transform
	2) lag preparation (effects selection influences LagEngine target rows)
	3) scaling (X and y)
	4) post-scaling load reduction (only to reduce training cost)
	5) optional hyperoptimization (short base-model training)
	6) full base model fit
	7) reference models with hot-start + omit_variables for each tested cause
	8) inverse-scale predictions and aggregation in GrangerAnalysisResults
	"""

	def __init__(
		self,
		backend: Optional[str] = None,
		reuse_data: bool = False,
	) -> None:
		self.backend = backend
		self._reuse_data = bool(reuse_data)
		self._prepared_data: Optional[PreparedData] = None

	

	def fit(
		self,
		data: pd.DataFrame | Sequence[pd.DataFrame],
		causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		tested_causes: Optional[Sequence[str]] = None,
		relations: Optional[Mapping[Tuple[str, str], Any]] = None,
		lag_config: Optional[LagConfiguration] = None,
		lag_selector: Optional[BaseLagSelector] = None,
		stationarity_transformer: Optional[StationarityTransformer] = None,
		prepared_data: Optional[PreparedData] = None,
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
		x_scaler: Optional[Any] = "standard",
		y_scaler: Optional[Any] = "standard",
		regularizer: Optional[Any] = None,
		regularizer_spec: Optional[Dict[str, Any]] = None,
		callbacks: Optional[Sequence[Any]] = None,
		hiperoptimalization_state: Optional[str] = None,
		hiperoptimalization_conf: Optional[Dict[str, Any]] = None,
		initializer: Optional[Any] = None,
		model_config: Optional[Dict[str, Any]] = None,
	) -> GrangerAnalysisResults:

		relations_map = dict(relations or {})
		preparation_time_seconds = 0.0
		if prepared_data is not None:
			prepared = prepared_data
		else:
			prep_start = time.perf_counter()
			prepared = prepare_data(
				data=data,
				causes=causes,
				tested_causes=tested_causes.copy(),
				effects=effects.copy(),
				lag_config=lag_config,
				lag_selector=lag_selector,
				stationarity_transformer=stationarity_transformer,
				backend_sample_fraction=backend_sample_fraction,
				backend_max_samples=backend_max_samples,
				x_scaler=x_scaler,
				y_scaler=y_scaler,
			)
			preparation_time_seconds = time.perf_counter() - prep_start

		strategy = BackendFactory.get_strategy(self.backend)

		reg_spec_base = dict(regularizer_spec or {})
		if not reg_spec_base and regularizer is not None:
			derived_regularizer_spec = _regularizer_spec_from_value(regularizer)
			if derived_regularizer_spec is not None:
				reg_spec_base = derived_regularizer_spec
		if str(reg_spec_base.get("type", "")).lower() == "lag_dependent_l1":
			lag_block_sizes = list((prepared.col_offsets[1:] - prepared.col_offsets[:-1]).astype(int))
			reg_spec_base.setdefault("max_lags_per_pred", lag_block_sizes)
			reg_spec_base.setdefault("col_offsets", list(prepared.col_offsets[:-1].astype(int)))


		constraint_obj = strategy.build_constraint_from_relations(
			relations=relations_map,
			predictor_names=prepared.causes,
			output_names=prepared.effects,
			col_offsets=prepared.col_offsets[:-1],
			n_features=prepared.X_scaled.shape[1],
			base_mask=prepared.base_mask,
		)

		results = GrangerAnalysisResults(
			effects=prepared.effects, 
			causes=prepared.tested_causes, 
			mask=constraint_obj.mask if hasattr(constraint_obj,"mask") else prepared.base_mask, 
			col_offsets=prepared.col_offsets, 
			y_true=prepared.y_raw,
			X_true=prepared.X_raw,
		)


		model_cfg = dict(model_config or {})
		callbacks_cfg = model_cfg.pop("callbacks", None)
		callbacks_template: Optional[List[Any]] = None
		if callbacks is not None:
			callbacks_template = list(callbacks)
		elif callbacks_cfg is not None:
			callbacks_template = list(callbacks_cfg)

		def _run_cfg(run_name: str, *, base_cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
			run_cfg = dict(base_cfg or model_cfg)
			run_callbacks = _build_callbacks_for_run(callbacks_template, run_name)
			if run_callbacks is not None:
				run_cfg["callbacks"] = run_callbacks
			return run_cfg

		initializer_weights: Optional[List[np.ndarray]] = None
		if initializer is not None:
			initializer_weights = self._build_initializer_weights(
				initializer=initializer,
				X_train=prepared.X_scaled,
				y_train=prepared.y_scaled,
				mask=prepared.base_mask,
			)

		# Optional hyperoptimization before full base training.
		if hiperoptimalization_conf is not None or hiperoptimalization_state is not None:
			hopt_conf_effective = dict(hiperoptimalization_conf or {})
			if hiperoptimalization_state is not None and "type" not in hopt_conf_effective and "mode" not in hopt_conf_effective:
				hopt_conf_effective["type"] = hiperoptimalization_state

			hyper_optimizer = MultiTaskGrangerHyperparameterOptimizer(strategy=strategy)
			hopt_result = hyper_optimizer.optimize(
				config=hopt_conf_effective,
				regularizer_spec=reg_spec_base,
				model_config=model_cfg,
				prepared=prepared,
				constraint_obj=constraint_obj,
				initializer_weights=initializer_weights,
				run_cfg_factory=_run_cfg,
				assign_initializer_weights=self._assign_initializer_weights,
			)
			hopt_applied = hopt_result.apply_to(
				model_config=model_cfg,
				regularizer_spec=reg_spec_base,
			)
			reg_spec_base = hopt_applied["regularizer_spec"]
			model_cfg = hopt_applied["model_config"]

		if str(reg_spec_base.get("type", "")).lower() == "lag_dependent_l1":
			lag_block_sizes = list((prepared.col_offsets[1:] - prepared.col_offsets[:-1]).astype(int))
			reg_spec_base.setdefault("max_lags_per_pred", lag_block_sizes)
			reg_spec_base.setdefault("col_offsets", list(prepared.col_offsets[:-1].astype(int)))

		reg_obj = regularizer
		if reg_spec_base:
			reg_obj = strategy.build_regularizer(reg_spec_base)

		base_cfg = _run_cfg("base_model")
		base_model = strategy.build_model(
			n_features=prepared.X_scaled.shape[1],
			n_outputs=prepared.y_scaled.shape[1],
			regularizer=reg_obj,
			constraint=constraint_obj,
			**base_cfg,
		)
		base_model.initialize(prepared.X_scaled, targets=prepared.y_scaled)

		if initializer_weights is not None:
			self._assign_initializer_weights(
				model=base_model,
				weights=initializer_weights,
			)

		base_model.fit()
		base_weights = base_model.get_weights()

		base_pred_scaled = np.asarray(base_model.predict(prepared.X_scaled), dtype=np.float64)
		base_pred_real = prepared.y_scaler.inverse_transform(base_pred_scaled)
		base_weight_matrix = np.asarray(base_weights[0], dtype=np.float64)

		

		results.set_base_snapshot( 
			predictions=base_pred_real,
			weights=base_weight_matrix,
			metadata={"backend": self.backend},
		)

		# Reuse a single reference model object; reinitialize state for each cause.
		reference_cfg = _run_cfg("reference_model_template")
		reference_model = strategy.build_model(
			n_features=prepared.X_scaled.shape[1],
			n_outputs=prepared.y_scaled.shape[1],
			regularizer=reg_obj,
			constraint=constraint_obj,
			**reference_cfg,
		)
		reference_needs_reinit = bool(getattr(reference_model, "needs_reinit", True))
		if not reference_needs_reinit:
			reference_model.initialize(prepared.X_scaled, targets=prepared.y_scaled)

		for cause_name in prepared.tested_causes:
			cause_idx = prepared.tested_causes.index(cause_name)

			reference_model.reset_callbacks(run_name=f"reference_cause_{cause_name}")

			if reference_needs_reinit:
				reference_model.initialize(prepared.X_scaled, targets=prepared.y_scaled)
			else:
				reference_model.set_weights(base_weights)

			start = int(prepared.col_offsets[cause_idx])
			end = int(prepared.col_offsets[cause_idx + 1])
			reference_model.omit_variables(list(range(start, end)))
			reference_model.fit()

			ref_pred_scaled = reference_model.predict(prepared.X_scaled)
			ref_pred_real = prepared.y_scaler.inverse_transform(ref_pred_scaled)
			ref_weight_matrix = reference_model.get_weights()[0]

			results.set_reference_snapshot(
				cause=cause_name,
				reference_predictions=ref_pred_real,
				reference_weights=ref_weight_matrix,
				metadata={"backend": self.backend}
			)

		results.set_preparation_context(
			PreparationContext(
				prepared_data=prepared if (self._reuse_data or prepared_data is not None) else None,
				metadata={
					"preparation_time_seconds": preparation_time_seconds,
				},
			)
		)

		return results

	@staticmethod
	def _build_initializer_weights(
		initializer: Any,
		X_train: np.ndarray,
		y_train: np.ndarray,
		mask: Optional[np.ndarray],
	) -> List[np.ndarray]:
		"""Compute initializer weights once and return [kernel, bias]."""
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
		return [kernel, B_arr]

	@staticmethod
	def _assign_initializer_weights(model: Any, weights: List[np.ndarray]) -> None:
		"""Assign precomputed initializer weights to a model."""
		kernel = np.asarray(weights[0], dtype=np.float64).copy()
		bias = np.asarray(weights[1], dtype=np.float64).copy()
		model.set_weights([kernel, bias])

	