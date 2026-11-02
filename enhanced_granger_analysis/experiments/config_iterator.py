from __future__ import annotations

import copy

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..configuration.loader import AnalysisConfigLoader
from ..core.exceptions import DataValidationError

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

	def __init__(self, configs: Sequence[Dict[str, Any]], base_dir: Optional[Path] = None) -> None:
		self._configs: List[Dict[str, Any]] = [copy.deepcopy(c) for c in configs]
		self._index = 0
		self._base_dir = Path(base_dir).resolve() if base_dir is not None else None

	@classmethod
	def from_file(
		cls,
		path: str | Path,
	) -> "TestGroupConfigIterator":
		"""Read a group document and expand its configured cases."""

		group_path = Path(path).expanduser().resolve()

		raw = AnalysisConfigLoader.load_raw_file(group_path)
		configs = cls._expand(raw)

		return cls(
			configs,
			base_dir=group_path.parent,
		)

	def __iter__(self) -> "TestGroupConfigIterator":
		"""Return self as the iterator object (implements iterator protocol)."""
		return self

	def __next__(self) -> Dict[str, Any]:
		"""Return one configuration with resolved document references."""

		if self._index >= len(self._configs):
			raise StopIteration

		config = copy.deepcopy(self._configs[self._index])

		if self._base_dir is not None:
			config = AnalysisConfigLoader.resolve_references(
				config,
				base_dir=self._base_dir,
			)

		self._index += 1

		return config

	@staticmethod
	def _expand(group_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
		base = dict(group_cfg.get("base_config", {}))
		group_meta = {
			key: copy.deepcopy(value)
			for key, value in group_cfg.items()
			if key not in {"base_config", "sweep"}
		}
		sweep = group_cfg.get("sweep")

		if sweep is None:
			merged = dict(group_meta)
			merged.update(base)
			return [merged]
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

			cfg = dict(group_meta)
			cfg.update(copy.deepcopy(base))
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
