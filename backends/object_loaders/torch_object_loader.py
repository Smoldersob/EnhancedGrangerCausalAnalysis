from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..callbacks import ConvergenceCheck, EarlyStopping, ReduceLearningRate
from ..callbacks.base_callback import Callback
from ..constraints.pytorch_constraints import PyTorchMaskAndMinAbsSumConstraint, PyTorchMaskConstraint
from ..regularizers.pytorch_regularizers import PyTorchL1Regularizer, PyTorchLagDependentL1Regularizer
from ...core.constraints_config import MinAbsSumRule, ProcessedConstraintSpec
from ...core.exceptions import ConstraintConfigurationError

try:  # pragma: no cover - optional dependency
	from ..callbacks import TorchTensorBoardCallback
except Exception:  # pragma: no cover - optional dependency
	TorchTensorBoardCallback = None


class TorchObjectLoader:
	"""Resolve PyTorch callback and optimizer specs."""

	def __init__(self, torch_module: Any, loading_verbose: bool = False) -> None:
		self._torch = torch_module
		self._loading_verbose = bool(loading_verbose)

	def set_loading_verbose(self, value: bool) -> None:
		self._loading_verbose = bool(value)

	def _log(self, label: str, value: Any) -> None:
		if self._loading_verbose:
			print(f"[BackendLoader] TorchObjectLoader: {label} -> {value!r}")

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
		if type_name in {"torch_tensorboard", "tensorboard", "tensorboard_logger"}:
			if TorchTensorBoardCallback is None:
				raise ConstraintConfigurationError(
					"Callback 'torch_tensorboard' is unavailable. Install PyTorch to enable it."
				)
			resolved = TorchTensorBoardCallback(**params)
			self._log("callback", resolved)
			return resolved

		raise ConstraintConfigurationError(
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
			resolved = self._torch.optim.Adam
			self._log("optimizer", resolved)
			return resolved

		if isinstance(raw_optimizer, str):
			mapping = {
				"adam": self._torch.optim.Adam,
				"sgd": self._torch.optim.SGD,
				"rmsprop": self._torch.optim.RMSprop,
			}
			resolved = mapping.get(raw_optimizer.strip().lower())
			if resolved is None:
				raise ConstraintConfigurationError(
					f"Unsupported optimizer '{raw_optimizer}'. Use one of: {sorted(mapping.keys())}"
				)
			self._log("optimizer", resolved)
			return resolved

		if isinstance(raw_optimizer, Mapping):
			type_name, params = self._extract_typed_spec(raw_optimizer, context="optimizer")
			opt_cls = self.resolve_optimizer(type_name)

			def _optimizer_factory(model_params: Any, lr: float = 1e-3, _cls: Any = opt_cls, _params: Dict[str, Any] = params) -> Any:
				merged = dict(_params)
				merged.setdefault("lr", lr)
				return _cls(model_params, **merged)

			self._log("optimizer(factory)", _optimizer_factory)
			return _optimizer_factory

		if isinstance(raw_optimizer, type) and issubclass(raw_optimizer, self._torch.optim.Optimizer):
			self._log("optimizer(class)", raw_optimizer)
			return raw_optimizer

		if callable(raw_optimizer):
			self._log("optimizer(callable)", raw_optimizer)
			return raw_optimizer

		raise ConstraintConfigurationError(
			"optimizer must be string, optimizer config object, optimizer class, or callable"
		)

	def resolve_regularizer(self, raw_regularizer: Any) -> Any:
		if raw_regularizer is None:
			return None

		if isinstance(raw_regularizer, (PyTorchL1Regularizer, PyTorchLagDependentL1Regularizer)):
			self._log("regularizer(instance)", raw_regularizer)
			return raw_regularizer

		if not isinstance(raw_regularizer, Mapping):
			return raw_regularizer

		type_name, params = self._extract_typed_spec(raw_regularizer, context="regularizer")
		if type_name in {"l1", "torch_l1", "pytorch_l1"}:
			resolved = PyTorchL1Regularizer(l1=float(params.get("l1", 0.01)))
			self._log("regularizer", resolved)
			return resolved
		if type_name in {"lag_dependent_l1", "lagdependentl1", "torch_lag_dependent_l1", "pytorch_lag_dependent_l1"}:
			resolved = PyTorchLagDependentL1Regularizer(
				l1=float(params.get("l1", 0.01)),
				lag_weights=params.get("lag_weights", None),
				max_lags_per_pred=params.get("max_lags_per_pred", None),
				col_offsets=params.get("col_offsets", None),
			)
			self._log("regularizer", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported regularizer type for pytorch backend. Supported: l1, lag_dependent_l1"
		)

	@staticmethod
	def _build_processed_constraint_spec(spec_map: Mapping[str, Any]) -> ProcessedConstraintSpec:
		mask = spec_map.get("mask")
		if mask is None:
			raise ConstraintConfigurationError("constraint spec requires 'mask'")

		raw_rules = spec_map.get("rules", [])
		if raw_rules is None:
			raw_rules = []
		if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
			raise ConstraintConfigurationError("constraint spec 'rules' must be an array")

		rules: List[MinAbsSumRule] = []
		for rule in raw_rules:
			if not isinstance(rule, Mapping):
				raise ConstraintConfigurationError("each constraint rule must be an object")
			rules.append(
				MinAbsSumRule(
					output_index=int(rule.get("output_index")),
					feature_indices=tuple(int(v) for v in rule.get("feature_indices", [])),
					min_abs_sum=float(rule.get("min_abs_sum", 0.0)),
				)
			)

		return ProcessedConstraintSpec(mask=mask, rules=tuple(rules))

	def resolve_constraint(self, raw_constraint: Any) -> Any:
		if raw_constraint is None:
			return None

		if isinstance(raw_constraint, (PyTorchMaskConstraint, PyTorchMaskAndMinAbsSumConstraint)):
			self._log("constraint(instance)", raw_constraint)
			return raw_constraint

		if not isinstance(raw_constraint, Mapping):
			return raw_constraint

		type_name, params = self._extract_typed_spec(raw_constraint, context="constraint")
		if type_name in {"mask", "mask_constraint", "torch_mask", "pytorch_mask"}:
			if "mask" not in params:
				raise ConstraintConfigurationError("mask constraint requires 'mask'")
			resolved = PyTorchMaskConstraint(mask=params["mask"])
			self._log("constraint", resolved)
			return resolved

		if type_name in {"mask_and_min_abs_sum", "mask_min_abs_sum", "torch_mask_and_min_abs_sum", "pytorch_mask_and_min_abs_sum"}:
			spec = self._build_processed_constraint_spec(params)
			resolved = PyTorchMaskAndMinAbsSumConstraint(spec=spec, eps=float(params.get("eps", 1e-8)))
			self._log("constraint", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported constraint type for pytorch backend. Supported: mask, mask_and_min_abs_sum"
		)
