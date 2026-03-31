from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..callbacks import ConvergenceCheck, EarlyStopping, ReduceLearningRate
from ..callbacks.base_callback import Callback


class NumpyObjectLoader:
	"""Resolve scikit/numpy callback specs and validate optimizer compatibility."""

	@staticmethod
	def _extract_typed_spec(raw_spec: Any, *, context: str) -> Tuple[str, Dict[str, Any]]:
		if isinstance(raw_spec, str):
			return raw_spec.strip().lower(), {}

		if not isinstance(raw_spec, Mapping):
			raise ValueError(f"{context} spec must be a string or object")

		type_name = raw_spec.get("type", raw_spec.get("name", raw_spec.get("kind")))
		if not isinstance(type_name, str) or not type_name.strip():
			raise ValueError(f"{context} spec requires a non-empty 'type' field")

		params_raw = raw_spec.get("params", {})
		if params_raw is None:
			params_raw = {}
		if not isinstance(params_raw, Mapping):
			raise ValueError(f"{context} spec 'params' must be an object")

		params = dict(params_raw)
		for key, value in raw_spec.items():
			if key not in {"type", "name", "kind", "params"}:
				params[key] = value

		return type_name.strip().lower(), params

	def resolve_callback(self, raw_callback: Any) -> Callback:
		if isinstance(raw_callback, Callback):
			return raw_callback

		type_name, params = self._extract_typed_spec(raw_callback, context="callback")

		if type_name in {"early_stopping", "earlystopping"}:
			return EarlyStopping(**params)
		if type_name in {"reduce_lr", "reduce_learning_rate", "reducelearningrate"}:
			return ReduceLearningRate(**params)
		if type_name in {"convergence_check", "convergencecheck"}:
			return ConvergenceCheck(**params)

		raise ValueError(
			"Unsupported callback type. Supported: early_stopping, reduce_lr, convergence_check"
		)

	def resolve_callbacks(self, callbacks_cfg: Optional[Sequence[Any]]) -> List[Callback]:
		if callbacks_cfg is None:
			return []

		if not isinstance(callbacks_cfg, Sequence) or isinstance(callbacks_cfg, (str, bytes)):
			callbacks_cfg = [callbacks_cfg]  # type: ignore[assignment]

		return [self.resolve_callback(cb) for cb in callbacks_cfg]

	def resolve_optimizer(self, raw_optimizer: Any) -> None:
		if raw_optimizer is None:
			return None

		raise ValueError(
			"Scikit backend does not support optimizer objects. Use learning_rate/max_iter/tol in model_config."
		)
