from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from pathlib import Path

import pandas as pd

from ..core.exceptions import DataValidationError
from ..core.lag_config import LagConfiguration
from ..core.outputs import MultitaskGrangerOutput
from ..preprocessing.stationarity import StationarityTransformer
from .config_loader import BuilderConfigLoader
from .orchestrator import MultiTaskGrangerAPI


class MultitaskGrangerBuilder:
	"""
	Fluent builder for MultiTaskGrangerAPI.

	The builder stores configuration step by step and executes analysis via fit().
	"""

	def __init__(self, backend: Optional[str] = None) -> None:
		self._backend: Optional[str] = backend
		self._data: Optional[pd.DataFrame | Sequence[pd.DataFrame]] = None

		self._fit_kwargs: Dict[str, Any] = {
			"causes": None,
			"effects": None,
			"tested_causes": None,
			"relations": None,
			"lag_config": None,
			"lag_selector": None,
			"stationarity_transformer": None,
			"backend_sample_fraction": 1.0,
			"backend_max_samples": None,
			"x_scaler": "standard",
			"y_scaler": "standard",
			"regularizer": None,
			"regularizer_spec": None,
			"callbacks": None,
			"hiperoptimalization_state": None,
			"hiperoptimalization_conf": None,
			"initializer": None,
			"model_config": None,
		}

	def backend(self, backend_name: Optional[str]) -> "MultitaskGrangerBuilder":
		self._backend = backend_name
		return self

	def data(self, data: pd.DataFrame | Sequence[pd.DataFrame]) -> "MultitaskGrangerBuilder":
		self._data = data
		return self

	def variables(
		self,
		causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		tested_causes: Optional[Sequence[str]] = None,
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["causes"] = list(causes) if causes is not None else None
		self._fit_kwargs["effects"] = list(effects) if effects is not None else None
		self._fit_kwargs["tested_causes"] = list(tested_causes) if tested_causes is not None else None
		return self

	def relations(
		self,
		relations: Optional[Mapping[Tuple[str, str], Any]],
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["relations"] = dict(relations) if relations is not None else None
		return self

	def lag(
		self,
		lag_config: Optional[LagConfiguration] = None,
		lag_selector: Optional[Any] = None,
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["lag_config"] = lag_config
		self._fit_kwargs["lag_selector"] = lag_selector
		return self

	def stationarity(
		self,
		transformer: Optional[StationarityTransformer] = None,
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["stationarity_transformer"] = transformer
		return self

	def scaling(
		self,
		x_scaler: Optional[str] = "standard",
		y_scaler: Optional[str] = "standard",
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["x_scaler"] = x_scaler
		self._fit_kwargs["y_scaler"] = y_scaler
		return self

	def backend_load(
		self,
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["backend_sample_fraction"] = float(backend_sample_fraction)
		self._fit_kwargs["backend_max_samples"] = backend_max_samples
		return self

	def regularization(
		self,
		regularizer: Optional[Any] = None,
		regularizer_spec: Optional[Dict[str, Any]] = None,
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["regularizer"] = regularizer
		self._fit_kwargs["regularizer_spec"] = dict(regularizer_spec) if regularizer_spec is not None else None
		return self

	def callbacks(self, callbacks: Optional[Sequence[Any]] = None) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["callbacks"] = list(callbacks) if callbacks is not None else None
		return self

	def hyperoptimization(
		self,
		state: Optional[str] = None,
		config: Optional[Dict[str, Any]] = None,
	) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["hiperoptimalization_state"] = state
		self._fit_kwargs["hiperoptimalization_conf"] = dict(config) if config is not None else None
		return self

	def initializer(self, initializer: Optional[Any]) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["initializer"] = initializer
		return self

	def model(self, model_config: Optional[Dict[str, Any]] = None) -> "MultitaskGrangerBuilder":
		self._fit_kwargs["model_config"] = dict(model_config) if model_config is not None else None
		return self

	def from_config(self, config: Mapping[str, Any]) -> "MultitaskGrangerBuilder":
		"""Load builder state from a config mapping using orchestrator fit keys."""
		if "backend" in config:
			self._backend = config.get("backend")

		if "data" in config:
			self._data = config.get("data")

		for key in self._fit_kwargs.keys():
			if key in config:
				value = config[key]
				if key in {"regularizer_spec", "hiperoptimalization_conf", "model_config"} and value is not None:
					self._fit_kwargs[key] = dict(value)
				elif key in {"causes", "effects", "tested_causes"} and value is not None:
					self._fit_kwargs[key] = list(value)
				elif key == "relations" and value is not None:
					self._fit_kwargs[key] = dict(value)
				else:
					self._fit_kwargs[key] = value

		return self

	def from_file(self, path: str | Path) -> "MultitaskGrangerBuilder":
		"""Load builder state from a JSON/YAML config file."""
		cfg = BuilderConfigLoader.load_file(path)
		return self.from_config(cfg)

	def fit(self) -> MultitaskGrangerOutput:
		if self._data is None:
			raise DataValidationError("Builder requires data(...) before fit()")

		api = MultiTaskGrangerAPI(backend=self._backend)
		return api.fit(self._data, **self._fit_kwargs)

	def run(self) -> MultitaskGrangerOutput:
		"""Alias for fit()."""
		return self.fit()


GrangerAnalysisBuilder = MultitaskGrangerBuilder


__all__ = ["MultitaskGrangerBuilder", "GrangerAnalysisBuilder"]
