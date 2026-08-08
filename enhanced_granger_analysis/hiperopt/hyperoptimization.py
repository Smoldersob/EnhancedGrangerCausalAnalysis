from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class HyperoptimizationResult:
	model_config_updates: Dict[str, Any] = field(default_factory=dict)
	regularizer_spec_updates: Dict[str, Any] = field(default_factory=dict)
	best_score: float = float("inf")
	best_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	state: Optional[str] = None

	def apply_to(
		self,
		*,
		model_config: Optional[Mapping[str, Any]] = None,
		regularizer_spec: Optional[Mapping[str, Any]] = None,
	) -> Dict[str, Any]:
		"""Merge the best sweep result into the current training state."""
		updated_model_config = dict(model_config or {})
		updated_model_config.update(self.model_config_updates)

		updated_regularizer_spec = dict(regularizer_spec or {})
		if self.regularizer_spec_updates:
			updated_regularizer_spec.update(self.regularizer_spec_updates)

		return {
			"model_config": updated_model_config,
			"regularizer_spec": updated_regularizer_spec,
			"best_params": dict(self.best_params),
		}


class MultiTaskGrangerHyperparameterOptimizer:
	"""Choose hyperparameters for multitask Granger analysis runs."""

	_SEARCH_SECTIONS = ("model", "optimizer", "regularizer")

	def __init__(self, strategy: Any) -> None:
		self._strategy = strategy

	def __expand_grid(self, param_grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
		if not param_grid:
			return []
		keys = list(param_grid.keys())
		values = [list(param_grid[k]) for k in keys]
		if any(len(v) == 0 for v in values):
			raise ValueError("All sweep entries must contain at least one value")
		return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

	def __extract_score(self, fit_result: Any) -> float:
		if isinstance(fit_result, dict) and "test_statistic" in fit_result:
			try:
				return float(fit_result["test_statistic"])
			except (TypeError, ValueError):
				pass
		return float("inf")

	def __default_section(self, state: Optional[str]) -> str:
		if state == "regularization":
			return "regularizer"
		return "model"

	def __partition_search_space(
		self,
		sweep_spec: Mapping[str, Any],
		*,
		default_section: str,
	) -> Dict[str, Dict[str, Sequence[Any]]]:
		section_grids: Dict[str, Dict[str, Sequence[Any]]] = {
			section: {} for section in self._SEARCH_SECTIONS
		}

		def _assign(section: str, param_name: str, values: Any) -> None:
			if section in section_grids:
				if isinstance(values, Sequence) and not isinstance(values, (str, bytes, Mapping)):
					section_grids[section][param_name] = values
				else:
					section_grids[section][param_name] = [values]

		def _visit(mapping: Mapping[str, Any], current_default: str) -> None:
			for key, value in mapping.items():
				if key in self._SEARCH_SECTIONS and isinstance(value, Mapping):
					_visit(value, key)
					continue

				if isinstance(key, str) and "." in key:
					section_name, param_name = key.split(".", 1)
					if section_name in self._SEARCH_SECTIONS:
						_assign(section_name, param_name, value)
						continue

				_assign(current_default, key, value)

		_visit(sweep_spec, default_section)
		return {section: grid for section, grid in section_grids.items() if grid}

	def __trial_train_config(
		self,
		*,
		base_model_config: Mapping[str, Any],
		raw_hopt_conf: Mapping[str, Any],
	) -> Dict[str, Any]:
		trial_cfg: Dict[str, Any] = {}
		if "epochs" in base_model_config:
			trial_cfg["epochs"] = max(3, int(base_model_config["epochs"] // 5))
		if "max_iter" in base_model_config:
			trial_cfg["max_iter"] = max(10, int(base_model_config["max_iter"] // 5))

		raw_trial_cfg = raw_hopt_conf.get("trial_train_config", {})
		if isinstance(raw_trial_cfg, Mapping):
			trial_cfg.update(dict(raw_trial_cfg))

		if "trial_epochs" in raw_hopt_conf:
			trial_cfg["epochs"] = int(raw_hopt_conf["trial_epochs"])

		return trial_cfg

	def __candidate_groups(
		self,
		section_grids: Mapping[str, Mapping[str, Sequence[Any]]],
	) -> Dict[str, Sequence[Dict[str, Any]]]:
		candidate_groups: Dict[str, Sequence[Dict[str, Any]]] = {}
		for section_name in self._SEARCH_SECTIONS:
			section_grid = section_grids.get(section_name, {})
			section_candidates = self.__expand_grid(section_grid) if section_grid else []
			candidate_groups[section_name] = section_candidates or [{}]
		return candidate_groups

	def optimize(
		self,
		*,
		config: Optional[Mapping[str, Any]],
		regularizer_spec: Optional[Mapping[str, Any]],
		model_config: Mapping[str, Any],
		prepared: Any,
		constraint_obj: Any,
		initializer_weights: Optional[Sequence[np.ndarray]],
		run_cfg_factory: Callable[[str], Dict[str, Any]],
		assign_initializer_weights: Callable[[Any, Sequence[np.ndarray]], None],
	) -> HyperoptimizationResult:
		hopt_conf = dict(config or {})
		if not hopt_conf:
			return HyperoptimizationResult(
				model_config_updates={},
				regularizer_spec_updates={},
				state=None,
			)

		hopt_state = hopt_conf.pop("type", hopt_conf.pop("mode", None))
		if hopt_state is None:
			hopt_state = "model"

		if hopt_state not in {"model", "regularization"}:
			raise ValueError("hiperoptimalization_state must be one of: None, 'model', 'regularization'")

		hopt_n_trials = int(hopt_conf.get("n_trials", 20))
		hopt_grid_source = hopt_conf.get("sections", hopt_conf.get("param_grid", {}))
		if not isinstance(hopt_grid_source, Mapping):
			hopt_grid_source = {}
		hopt_default_section = self.__default_section(hopt_state)
		hopt_section_grids = self.__partition_search_space(
			hopt_grid_source,
			default_section=hopt_default_section,
		)

		base_model_config = dict(model_config)
		base_regularizer_spec = dict(regularizer_spec or {})
		trial_train_cfg = self.__trial_train_config(
			base_model_config=base_model_config,
			raw_hopt_conf=hopt_conf,
		)

		section_candidate_groups = self.__candidate_groups(hopt_section_grids)

		best_score = float("inf")
		best_regularizer_spec = dict(base_regularizer_spec)
		best_model_config = dict(base_model_config)
		best_params: Dict[str, Dict[str, Any]] = {}

		trial_iter = itertools.product(
			section_candidate_groups["model"],
			section_candidate_groups["optimizer"],
			section_candidate_groups["regularizer"],
		)
		for trial_idx, (model_params, optimizer_params, regularizer_params) in enumerate(trial_iter, start=1):
			if trial_idx > hopt_n_trials:
				break

			trial_model_config = dict(base_model_config)
			trial_model_config.update(trial_train_cfg)
			trial_model_config.update(model_params)
			trial_model_config.update(optimizer_params)

			trial_reg_spec = dict(base_regularizer_spec)
			trial_reg_spec.update(regularizer_params)
			trial_reg = None
			if trial_reg_spec:
				trial_reg = self._strategy.build_regularizer(trial_reg_spec)

			trial_cfg = run_cfg_factory(f"hopt_trial_{trial_idx}", base_cfg=trial_model_config)
			trial_model = self._strategy.build_model(
				n_features=prepared.X_train.shape[1],
				n_outputs=prepared.y_train.shape[1],
				regularizer=trial_reg,
				constraint=constraint_obj,
				**trial_cfg,
			)
			trial_model.initialize(prepared.X_backend_scaled, targets=prepared.y_backend_scaled)
			if initializer_weights is not None:
				assign_initializer_weights(
					model=trial_model,
					weights=initializer_weights,
				)
			fit_result = trial_model.fit()
			score = self.__extract_score(fit_result)
			if score < best_score:
				best_score = score
				best_regularizer_spec = dict(trial_reg_spec)
				best_model_config = {
					k: v for k, v in trial_model_config.items() if k not in trial_train_cfg
				}
				best_params = {
					"model": dict(model_params),
					"optimizer": dict(optimizer_params),
					"regularizer": dict(regularizer_params),
				}

		return HyperoptimizationResult(
			model_config_updates=best_model_config,
			regularizer_spec_updates=best_regularizer_spec,
			best_score=best_score,
			best_params=best_params,
			state=hopt_state,
		)


__all__ = ["HyperoptimizationResult", "MultiTaskGrangerHyperparameterOptimizer"]
