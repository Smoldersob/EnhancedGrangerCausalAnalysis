from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..core.exceptions import DataValidationError
from ..core.lag_config import LagConfiguration


class BuilderConfigLoader:
	"""Load and normalize builder configuration from JSON/YAML files."""

	@staticmethod
	def load_raw_file(path: str | Path) -> Dict[str, Any]:
		"""Load raw mapping from JSON/YAML file without normalization."""
		return BuilderConfigLoader._read_mapping(path)

	@staticmethod
	def load_file(path: str | Path) -> Dict[str, Any]:
		raw = BuilderConfigLoader._read_mapping(path)
		return BuilderConfigLoader.normalize_builder_config(raw)

	@staticmethod
	def normalize_builder_config(config: Mapping[str, Any]) -> Dict[str, Any]:
		out = dict(config)

		lag_cfg = out.get("lag_config")
		if isinstance(lag_cfg, Mapping):
			out["lag_config"] = LagConfiguration(**dict(lag_cfg))

		return out

	@staticmethod
	def _read_mapping(path: str | Path) -> Dict[str, Any]:
		cfg_path = Path(path)
		if not cfg_path.exists():
			raise DataValidationError(f"Config file does not exist: {cfg_path}")

		suffix = cfg_path.suffix.lower()
		text = cfg_path.read_text(encoding="utf-8")

		if suffix == ".json":
			data = json.loads(text)
		elif suffix in {".yml", ".yaml"}:
			try:
				import yaml  # type: ignore
			except Exception as exc:  # pragma: no cover - optional runtime dependency
				raise DataValidationError(
					"YAML support requires PyYAML. Install with: pip install pyyaml"
				) from exc
			data = yaml.safe_load(text)
		else:
			raise DataValidationError(
				f"Unsupported config extension '{suffix}'. Use .json, .yml or .yaml"
			)

		if not isinstance(data, Mapping):
			raise DataValidationError("Top-level configuration must be a mapping/object")

		return dict(data)


class TestGroupConfigIterator:
	"""
	Expand a test-group config into concrete builder configs.

	Supported format:
	{
	  "base_config": {...},
	  "sweep": {
	    "param_names": ["model_config.epochs", "model_config.learning_rate"],
	    "cases": [[20, 0.001], [50, 0.0005]]
	  }
	}
	"""

	def __init__(self, configs: Sequence[Dict[str, Any]]) -> None:
		self._configs: List[Dict[str, Any]] = [copy.deepcopy(c) for c in configs]
		self._index = 0

	@classmethod
	def from_file(cls, path: str | Path) -> "TestGroupConfigIterator":
		raw = BuilderConfigLoader._read_mapping(path)
		configs = cls._expand(raw)
		return cls(configs)

	def has_next(self) -> bool:
		return self._index < len(self._configs)

	def next(self) -> Dict[str, Any]:
		if not self.has_next():
			raise StopIteration("No more test configurations")

		cfg = copy.deepcopy(self._configs[self._index])
		self._index += 1
		return BuilderConfigLoader.normalize_builder_config(cfg)

	@staticmethod
	def _expand(group_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
		base = dict(group_cfg.get("base_config", {}))
		sweep = group_cfg.get("sweep")

		if sweep is None:
			return [base]
		if not isinstance(sweep, Mapping):
			raise DataValidationError("'sweep' must be a mapping")

		param_names = sweep.get("param_names", [])
		cases = sweep.get("cases", [])

		if not isinstance(param_names, list) or not all(isinstance(n, str) for n in param_names):
			raise DataValidationError("'sweep.param_names' must be a list of strings")
		if not isinstance(cases, list):
			raise DataValidationError("'sweep.cases' must be a list")

		out: List[Dict[str, Any]] = []
		for idx, row in enumerate(cases):
			if not isinstance(row, list):
				raise DataValidationError(f"sweep.cases[{idx}] must be a list")
			if len(row) != len(param_names):
				raise DataValidationError(
					f"sweep.cases[{idx}] has length {len(row)}, expected {len(param_names)}"
				)

			cfg = copy.deepcopy(base)
			for key, value in zip(param_names, row):
				TestGroupConfigIterator._set_dotted(cfg, key, value)
			out.append(cfg)

		return out

	@staticmethod
	def _set_dotted(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
		parts = [p for p in dotted_key.split(".") if p]
		if not parts:
			raise DataValidationError("Empty parameter name in sweep")

		node: Dict[str, Any] = target
		for p in parts[:-1]:
			child = node.get(p)
			if child is None:
				node[p] = {}
				child = node[p]
			if not isinstance(child, dict):
				raise DataValidationError(
					f"Cannot set nested key '{dotted_key}': '{p}' is not a mapping"
				)
			node = child

		node[parts[-1]] = value
