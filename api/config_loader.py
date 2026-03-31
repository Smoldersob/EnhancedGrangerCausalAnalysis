from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.exceptions import DataValidationError
from ..core.lag_config import LagConfiguration
from ..preprocessing.lag.lag_selectors import CVLagSelector, ICLagSelector, VARLagSelector
from ..callbacks import ConvergenceCheck, EarlyStopping, ReduceLearningRate

try:
	from ..callbacks import TorchTensorBoardCallback  # type: ignore
except Exception:  # pragma: no cover - optional callback
	TorchTensorBoardCallback = None  # type: ignore[assignment]


_RELATION_KEY_SEPARATORS = ("->", "|", ",", ":")


def _parse_relation_key(key: Any) -> Tuple[str, str]:
	"""Parse relation key into (output_name, predictor_name)."""
	if isinstance(key, (list, tuple)) and len(key) == 2:
		out_name, in_name = key
		if not isinstance(out_name, str) or not isinstance(in_name, str):
			raise DataValidationError("Relation tuple/list key must contain strings")
		return out_name, in_name

	if isinstance(key, str):
		for sep in _RELATION_KEY_SEPARATORS:
			if sep in key:
				parts = [p.strip() for p in key.split(sep)]
				if len(parts) == 2 and all(parts):
					return parts[0], parts[1]
		raise DataValidationError(
			"String relation key must be in format 'effect->cause', 'effect|cause', 'effect,cause', or 'effect:cause'"
		)

	raise DataValidationError(
		"Relation key must be a 2-item list/tuple or a string key with a supported separator"
	)


def _normalize_relations_config(raw_relations: Any) -> Dict[Tuple[str, str], Any]:
	"""Normalize JSON-friendly relation constraints into tuple-key mapping."""
	if raw_relations is None:
		return {}

	if isinstance(raw_relations, Mapping):
		out: Dict[Tuple[str, str], Any] = {}
		for key, value in raw_relations.items():
			out[_parse_relation_key(key)] = value
		return out

	if isinstance(raw_relations, Sequence) and not isinstance(raw_relations, (str, bytes)):
		out: Dict[Tuple[str, str], Any] = {}
		for idx, item in enumerate(raw_relations):
			if not isinstance(item, Mapping):
				raise DataValidationError(f"relations[{idx}] must be an object")

			out_name = item.get("effect", item.get("output", item.get("target")))
			in_name = item.get("cause", item.get("predictor", item.get("input")))
			if not isinstance(out_name, str) or not isinstance(in_name, str):
				raise DataValidationError(
					f"relations[{idx}] must define string effect/output/target and cause/predictor/input"
				)

			if "rule" in item:
				raw_value = item["rule"]
			elif "value" in item:
				raw_value = item["value"]
			elif any(k in item for k in ("zero", "min_abs_sum", "force_abs_sum")):
				raw_value = {
					k: item[k]
					for k in ("zero", "min_abs_sum", "force_abs_sum")
					if k in item
				}
			else:
				raise DataValidationError(
					f"relations[{idx}] must contain one of: rule, value, zero, min_abs_sum, force_abs_sum"
				)

			out[(out_name, in_name)] = raw_value
		return out

	raise DataValidationError("relations must be a mapping or a list of relation objects")


def _extract_typed_spec(raw_spec: Any, *, context: str) -> Tuple[str, Dict[str, Any]]:
	"""Extract a lowercase type name and merged params from a spec mapping/string."""
	if isinstance(raw_spec, str):
		return raw_spec.strip().lower(), {}

	if not isinstance(raw_spec, Mapping):
		raise DataValidationError(f"{context} must be a string or object")

	type_name = raw_spec.get("type", raw_spec.get("name", raw_spec.get("kind")))
	if not isinstance(type_name, str) or not type_name.strip():
		raise DataValidationError(f"{context} requires a non-empty 'type' (or alias: name/kind)")

	params_raw = raw_spec.get("params", {})
	if params_raw is None:
		params_raw = {}
	if not isinstance(params_raw, Mapping):
		raise DataValidationError(f"{context}.params must be an object")

	params = dict(params_raw)
	for key, value in raw_spec.items():
		if key not in {"type", "name", "kind", "params"}:
			params[key] = value

	return type_name.strip().lower(), params


def _build_lag_selector_from_spec(
	raw_selector: Any,
	*,
	lag_config: Optional[LagConfiguration],
) -> Any:
	"""Instantiate supported built-in lag selectors from config spec."""
	type_name, params = _extract_typed_spec(raw_selector, context="lag_selector")

	if "max_lag" not in params:
		params["max_lag"] = lag_config.max_lag if lag_config is not None else 12
	if "use_lag_zero" not in params:
		params["use_lag_zero"] = lag_config.use_lag_zero if lag_config is not None else False

	if type_name in {"ic", "iclagselector", "aic", "bic"}:
		if type_name == "bic":
			params.setdefault("use_bic", True)
		return ICLagSelector(**params)

	if type_name in {"cv", "cvlagselector", "cross_validation", "cross-validation"}:
		return CVLagSelector(**params)

	if type_name in {"var", "varlagselector"}:
		return VARLagSelector(**params)

	raise DataValidationError(
		"Unsupported lag_selector type. Supported: ic, cv, var"
	)


def _build_callback_from_spec(raw_callback: Any) -> Any:
	"""Instantiate supported built-in callbacks from config spec."""
	type_name, params = _extract_typed_spec(raw_callback, context="callback")

	if type_name in {"early_stopping", "earlystopping"}:
		return EarlyStopping(**params)
	if type_name in {"reduce_lr", "reduce_learning_rate", "reducelearningrate"}:
		return ReduceLearningRate(**params)
	if type_name in {"convergence_check", "convergencecheck"}:
		return ConvergenceCheck(**params)
	if type_name in {"torch_tensorboard", "tensorboard", "tensorboard_logger"}:
		if TorchTensorBoardCallback is None:
			raise DataValidationError(
				"Callback 'torch_tensorboard' is unavailable. Install PyTorch to enable it."
			)
		return TorchTensorBoardCallback(**params)

	raise DataValidationError(
		"Unsupported callback type. Supported: early_stopping, reduce_lr, convergence_check, torch_tensorboard"
	)


def _normalize_callbacks(raw_callbacks: Any) -> List[Any]:
	"""Normalize callback config into callback object instances."""
	if raw_callbacks is None:
		return []

	if isinstance(raw_callbacks, (str, Mapping)):
		return [_build_callback_from_spec(raw_callbacks)]

	if isinstance(raw_callbacks, Sequence) and not isinstance(raw_callbacks, (str, bytes)):
		return [_build_callback_from_spec(cb) for cb in raw_callbacks]

	raise DataValidationError("callbacks must be a callback spec object/string or a list of specs")


def _normalize_callbacks_for_backend(raw_callbacks: Any, backend_name: Optional[str]) -> List[Any]:
	"""Normalize callbacks with backend-aware behavior.

	For TensorFlow backends, callback specs are passed through as raw string/object
	specifications and are instantiated in tensorflow_backend.
	For other backends, only built-in library callbacks are allowed and instantiated here.
	"""
	if raw_callbacks is None:
		return []

	backend_lower = (backend_name or "").strip().lower()
	is_tensorflow_backend = backend_lower in {"tensorflow", "tf", "keras"}

	if not is_tensorflow_backend:
		return _normalize_callbacks(raw_callbacks)

	if isinstance(raw_callbacks, (str, Mapping)):
		return [raw_callbacks]

	if isinstance(raw_callbacks, Sequence) and not isinstance(raw_callbacks, (str, bytes)):
		out: List[Any] = []
		for idx, cb in enumerate(raw_callbacks):
			if isinstance(cb, (str, Mapping)):
				out.append(cb)
			else:
				raise DataValidationError(
					f"callbacks[{idx}] for TensorFlow backend must be a string or object spec"
				)
		return out

	raise DataValidationError("callbacks must be a callback spec object/string or a list of specs")


def _normalize_regularizer_spec(raw_regularizer: Any) -> Dict[str, Any]:
	"""Normalize regularizer spec aliases into backend-ready regularizer_spec mapping."""
	type_name, params = _extract_typed_spec(raw_regularizer, context="regularizer")
	params = dict(params)
	params["type"] = type_name

	if params["type"] not in {"l1", "lag_dependent_l1"}:
		raise DataValidationError(
			"Unsupported regularizer type. Supported: l1, lag_dependent_l1"
		)

	return params


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
		backend_name = out.get("backend") if isinstance(out.get("backend"), str) else None

		lag_cfg = out.get("lag_config")
		if isinstance(lag_cfg, Mapping):
			out["lag_config"] = LagConfiguration(**dict(lag_cfg))
		elif lag_cfg is not None and not isinstance(lag_cfg, LagConfiguration):
			raise DataValidationError("lag_config must be an object or LagConfiguration instance")

		if "lag_selector" in out and out["lag_selector"] is not None:
			out["lag_selector"] = _build_lag_selector_from_spec(
				out["lag_selector"],
				lag_config=out.get("lag_config") if isinstance(out.get("lag_config"), LagConfiguration) else None,
			)

		if "relations" in out:
			out["relations"] = _normalize_relations_config(out.get("relations"))

		if "callbacks" in out:
			out["callbacks"] = _normalize_callbacks_for_backend(
				out.get("callbacks"),
				backend_name=backend_name,
			)

		if "regularizer" in out and out.get("regularizer") is not None:
			if "regularizer_spec" in out and out.get("regularizer_spec") is not None:
				raise DataValidationError(
					"Provide only one of 'regularizer' or 'regularizer_spec' in config"
				)
			out["regularizer_spec"] = _normalize_regularizer_spec(out.pop("regularizer"))

		if "regularizer_spec" in out and out.get("regularizer_spec") is not None:
			out["regularizer_spec"] = _normalize_regularizer_spec(out.get("regularizer_spec"))

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
