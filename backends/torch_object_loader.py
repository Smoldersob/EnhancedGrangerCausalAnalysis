from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..callbacks import ConvergenceCheck, EarlyStopping, ReduceLearningRate
from ..callbacks.base_callback import Callback

try:  # pragma: no cover - optional dependency
	from ..callbacks import TorchTensorBoardCallback
except Exception:  # pragma: no cover - optional dependency
	TorchTensorBoardCallback = None


class TorchObjectLoader:
	"""Resolve PyTorch callback and optimizer specs."""

	def __init__(self, torch_module: Any) -> None:
		self._torch = torch_module

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
		if type_name in {"torch_tensorboard", "tensorboard", "tensorboard_logger"}:
			if TorchTensorBoardCallback is None:
				raise ValueError(
					"Callback 'torch_tensorboard' is unavailable. Install PyTorch to enable it."
				)
			return TorchTensorBoardCallback(**params)

		raise ValueError(
			"Unsupported callback type. Supported: early_stopping, reduce_lr, convergence_check, torch_tensorboard"
		)

	def resolve_callbacks(self, callbacks_cfg: Optional[Sequence[Any]]) -> List[Callback]:
		if callbacks_cfg is None:
			return []

		if not isinstance(callbacks_cfg, Sequence) or isinstance(callbacks_cfg, (str, bytes)):
			callbacks_cfg = [callbacks_cfg]  # type: ignore[assignment]

		return [self.resolve_callback(cb) for cb in callbacks_cfg]

	def resolve_optimizer(self, raw_optimizer: Any) -> Any:
		if raw_optimizer is None:
			return self._torch.optim.Adam

		if isinstance(raw_optimizer, str):
			mapping = {
				"adam": self._torch.optim.Adam,
				"sgd": self._torch.optim.SGD,
				"rmsprop": self._torch.optim.RMSprop,
			}
			resolved = mapping.get(raw_optimizer.strip().lower())
			if resolved is None:
				raise ValueError(
					f"Unsupported optimizer '{raw_optimizer}'. Use one of: {sorted(mapping.keys())}"
				)
			return resolved

		if isinstance(raw_optimizer, Mapping):
			type_name, params = self._extract_typed_spec(raw_optimizer, context="optimizer")
			opt_cls = self.resolve_optimizer(type_name)

			def _optimizer_factory(model_params: Any, lr: float = 1e-3, _cls: Any = opt_cls, _params: Dict[str, Any] = params) -> Any:
				merged = dict(_params)
				merged.setdefault("lr", lr)
				return _cls(model_params, **merged)

			return _optimizer_factory

		if isinstance(raw_optimizer, type) and issubclass(raw_optimizer, self._torch.optim.Optimizer):
			return raw_optimizer

		if callable(raw_optimizer):
			return raw_optimizer

		raise ValueError(
			"optimizer must be string, optimizer config object, optimizer class, or callable"
		)
