from __future__ import annotations

import copy
import itertools
import os
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

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


def _expand_grid(param_grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
	"""Expand parameter grid into cartesian product list."""
	if not param_grid:
		return []
	keys = list(param_grid.keys())
	values = [list(param_grid[k]) for k in keys]
	if any(len(v) == 0 for v in values):
		raise ValueError("All param_grid entries must contain at least one value")
	return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _extract_score_from_fit_result(fit_result: Any) -> float:
	"""Return scalar score used by hyperoptimization (lower is better)."""
	if isinstance(fit_result, dict) and "test_statistic" in fit_result:
		try:
			return float(fit_result["test_statistic"])
		except (TypeError, ValueError):
			pass
	return float("inf")


def _short_training_config(model_config: Mapping[str, Any]) -> Dict[str, Any]:
	"""Build a short-run config for hyperoptimization trial training."""
	short_cfg = dict(model_config)
	if "epochs" in short_cfg:
		short_cfg["epochs"] = max(3, int(short_cfg["epochs"] // 5))
	if "max_iter" in short_cfg:
		short_cfg["max_iter"] = max(10, int(short_cfg["max_iter"] // 5))
	return short_cfg


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


def _set_model_callbacks(model: Any, callbacks: Optional[List[Any]]) -> None:
	"""Set callbacks on an already created model and validate when model supports it."""
	if not hasattr(model, "callbacks"):
		return

	setattr(model, "callbacks", callbacks or [])
	validate = getattr(model, "_validate_callbacks", None)
	if callable(validate):
		validate()


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
		callbacks: Optional[Sequence[Any]] = None,
		hiperoptimalization_state: Optional[Literal["model", "regularization"]] = None,
		hiperoptimalization_conf: Optional[Dict[str, Any]] = None,
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
		X_train, y_train, col_offsets = engine.prepare(data_stationary, effects=effects_list)

		x_scaler_obj = _create_scaler(x_scaler)
		y_scaler_obj = _create_scaler(y_scaler)

		X_train_scaled = x_scaler_obj.fit_transform(X_train)
		y_train_scaled = y_scaler_obj.fit_transform(y_train)

		X_backend_scaled, y_backend_scaled, X_backend_raw, y_backend_raw = _reduce_backend_load(
			X_scaled=X_train_scaled,
			y_scaled=y_train_scaled,
			X_raw=X_train,
			y_raw=y_train,
			backend_sample_fraction=backend_sample_fraction,
			backend_max_samples=backend_max_samples,
		)

		strategy = BackendFactory.get_strategy(self.backend)

		reg_spec_base = dict(regularizer_spec or {})
		if str(reg_spec_base.get("type", "")).lower() == "lag_dependent_l1":
			lag_block_sizes = list((col_offsets[1:] - col_offsets[:-1]).astype(int))
			reg_spec_base.setdefault("max_lags_per_pred", lag_block_sizes)
			reg_spec_base.setdefault("col_offsets", list(col_offsets[:-1].astype(int)))

		reg_obj = regularizer
		if reg_obj is None and reg_spec_base:
			reg_obj = strategy.build_regularizer(reg_spec_base)

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
			n_features=X_train.shape[1],
			base_mask=base_mask,
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

		# Optional hyperoptimization before full base training.
		hopt_state = hiperoptimalization_state
		hopt_conf = dict(hiperoptimalization_conf or {})
		if hopt_state is not None:
			hopt_n_trials = int(hopt_conf.get("n_trials", 20))
			hopt_grid = dict(hopt_conf.get("param_grid", hopt_conf))
			if "n_trials" in hopt_grid:
				hopt_grid.pop("n_trials")
			short_cfg = _short_training_config(model_cfg)

			if hopt_state == "regularization":
				candidates = _expand_grid(hopt_grid) if hopt_grid else []
				if not candidates:
					candidates = [{}]

				best_score = float("inf")
				best_reg_obj = reg_obj

				for trial_idx, params in enumerate(candidates[:hopt_n_trials], start=1):
					trial_reg_spec = dict(reg_spec_base)
					trial_reg_spec.update(params)
					trial_reg = regularizer if regularizer is not None else strategy.build_regularizer(trial_reg_spec)

					trial_cfg = _run_cfg(f"hopt_regularization_trial_{trial_idx}")
					trial_cfg.update(short_cfg)

					trial_model = strategy.build_model(
						n_features=X_train.shape[1],
						n_outputs=y_train.shape[1],
						regularizer=trial_reg,
						constraint=constraint_obj,
						scaler=None,
						**trial_cfg,
					)
					trial_model.initialize(X_backend_scaled, targets=y_backend_scaled)
					if initializer is not None:
						self._apply_initializer(
							model=trial_model,
							initializer=initializer,
							X_train=X_backend_scaled,
							y_train=y_backend_scaled,
							mask=base_mask,
						)
					fit_result = trial_model.fit()
					score = _extract_score_from_fit_result(fit_result)
					if score < best_score:
						best_score = score
						best_reg_obj = trial_reg

				reg_obj = best_reg_obj

			elif hopt_state == "model":
				probe_cfg = _run_cfg("hopt_model_probe")
				probe_cfg.update(short_cfg)
				probe_model = strategy.build_model(
					n_features=X_train.shape[1],
					n_outputs=y_train.shape[1],
					regularizer=reg_obj,
					constraint=constraint_obj,
					scaler=None,
					**probe_cfg,
				)
				probe_model.initialize(X_backend_scaled, targets=y_backend_scaled)
				if initializer is not None:
					self._apply_initializer(
						model=probe_model,
						initializer=initializer,
						X_train=X_backend_scaled,
						y_train=y_backend_scaled,
						mask=base_mask,
					)
				if hasattr(probe_model, "hyperoptimize"):
					hopt_result = probe_model.hyperoptimize(
						reg_param_grid=hopt_grid,
						n_trials=hopt_n_trials,
					)
					best_params = hopt_result.get("best_params", {}) if isinstance(hopt_result, dict) else {}
					if isinstance(best_params, dict):
						model_cfg.update(best_params)
			else:
				raise ValueError("hiperoptimalization_state must be one of: None, 'model', 'regularization'")

		base_cfg = _run_cfg("base_model")
		base_model = strategy.build_model(
			n_features=X_train.shape[1],
			n_outputs=y_train.shape[1],
			regularizer=reg_obj,
			constraint=constraint_obj,
			scaler=None,
			**base_cfg,
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
		base_weight_matrix = np.asarray(base_weights[0], dtype=np.float64)

		results = GrangerAnalysisResults(effects=effects_list, causes=tested_causes_list)
		reference_models: Dict[str, Any] = {}

		# Reuse a single reference model object; reinitialize state for each cause.
		reference_cfg = _run_cfg("reference_model_template")
		reference_model = strategy.build_model(
			n_features=X_train.shape[1],
			n_outputs=y_train.shape[1],
			regularizer=reg_obj,
			constraint=constraint_obj,
			scaler=None,
			**reference_cfg,
		)

		for cause_name in tested_causes_list:
			cause_idx = all_columns.index(cause_name)
			_set_model_callbacks(
				reference_model,
				_build_callbacks_for_run(callbacks_template, f"reference_cause_{cause_name}"),
			)

			reference_model.initialize(X_backend_scaled, targets=y_backend_scaled)
			reference_model.set_weights(base_weights)

			start = int(col_offsets[cause_idx])
			end = int(col_offsets[cause_idx + 1])
			reference_model.omit_variables(list(range(start, end)))
			reference_model.fit()

			ref_pred_scaled = np.asarray(reference_model.predict(X_backend_scaled), dtype=np.float64)
			ref_pred_real = y_scaler_obj.inverse_transform(ref_pred_scaled)
			ref_weight_matrix = np.asarray(reference_model.get_weights()[0], dtype=np.float64)

			results.update_cause(
				cause=cause_name,
				cause_index=cause_idx,
				base_model=None,
				reference_model=None,
				X=X_backend_raw,
				y=y_backend_real,
				col_offsets=col_offsets,
				base_predictions=base_pred_real,
				reference_predictions=ref_pred_real,
				base_weights=base_weight_matrix,
				reference_weights=ref_weight_matrix,
			)
			reference_models[cause_name] = {
				"predictions": ref_pred_real,
				"weights": ref_weight_matrix,
			}

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
