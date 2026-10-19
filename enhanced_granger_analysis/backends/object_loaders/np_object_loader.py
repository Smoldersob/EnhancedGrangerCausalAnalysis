from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..callbacks import ConvergenceCheck, EarlyStopping, ReduceLearningRate
from ..callbacks.base_callback import Callback
from ...core.exceptions import ConstraintConfigurationError


class NumpyObjectLoader:
	"""Resolve scikit/numpy callback specs and validate optimizer compatibility."""

	def __init__(self, loading_verbose: bool = False):
		self._loading_verbose = bool(loading_verbose)

	@staticmethod
	def resolve_model(model_name: Optional[str], default_model: Any) -> Any:
		"""Resolve a current backend model, accepting either a name or a subclass.

		Supported cases:
		- None / default names -> default_model
		- a subclass of default_model -> return that subclass
		- any other value -> keep the existing default to preserve backward compatibility
		"""
		if model_name is None:
			return default_model

		if isinstance(model_name, type) and issubclass(model_name, default_model):
			return model_name

		normalized = str(model_name).strip().lower()
		if normalized in {"", "default", "base", "standard", "linear", "granger"}:
			return default_model

		return default_model

	def set_loading_verbose(self, value: bool) -> None:
		self._loading_verbose = bool(value)

	def _log(self, label: str, value: Any) -> None:
		if self._loading_verbose:
			print(f"[BackendLoader] NumpyObjectLoader: {label} -> {value!r}")

	@staticmethod
	def _extract_typed_spec(raw_spec: Any, *, context: str) -> Tuple[str, Dict[str, Any]]:
		if isinstance(raw_spec, str):
			return raw_spec.strip().lower(), {}

		if not isinstance(raw_spec, Mapping):
			raise ConstraintConfigurationError(f"{context} spec must be a string or object")

		type_name = raw_spec.get("type", raw_spec.get("name", raw_spec.get("kind")))
		if not isinstance(type_name, str) or not type_name.strip():
			raise ConstraintConfigurationError(f"{context} spec requires a non-empty 'type' field")

		params_raw = raw_spec.get("params", {})
		if params_raw is None:
			params_raw = {}
		if not isinstance(params_raw, Mapping):
			raise ConstraintConfigurationError(f"{context} spec 'params' must be an object")

		params = dict(params_raw)
		for key, value in raw_spec.items():
			if key not in {"type", "name", "kind", "params"}:
				params[key] = value

		return type_name.strip().lower(), params

	def resolve_callback(self, raw_callback: Any) -> Callback:
		if isinstance(raw_callback, Callback):
			self._log("callback(instance)", raw_callback)
			return raw_callback

		type_name, params = self._extract_typed_spec(raw_callback, context="callback")

		if type_name in {"early_stopping", "earlystopping"}:
			resolved = EarlyStopping(**params)
			self._log("callback", resolved)
			return resolved
		if type_name in {"reduce_lr", "reduce_learning_rate", "reducelearningrate"}:
			resolved = ReduceLearningRate(**params)
			self._log("callback", resolved)
			return resolved
		if type_name in {"convergence_check", "convergencecheck"}:
			resolved = ConvergenceCheck(**params)
			self._log("callback", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported callback type. Supported: early_stopping, reduce_lr, convergence_check"
		)

	def resolve_callbacks(self, callbacks_cfg: Optional[Sequence[Any]]) -> List[Callback]:
		if callbacks_cfg is None:
			return []

		if not isinstance(callbacks_cfg, Sequence) or isinstance(callbacks_cfg, (str, bytes)):
			callbacks_cfg = [callbacks_cfg]  # type: ignore[assignment]

		return [self.resolve_callback(cb) for cb in callbacks_cfg]

	def resolve_optimizer(self, raw_optimizer: Any) -> None:
		return None

