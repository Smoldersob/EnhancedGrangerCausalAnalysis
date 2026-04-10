from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..callbacks import ConvergenceCheck, EarlyStopping, ReduceLearningRate
from ..callbacks.base_callback import Callback
from ..constraints.numpy_constraints import NumpyMaskAndMinAbsSumConstraint, NumpyMaskConstraint
from ..regularizers.numpy_regularizers import NumpyL1Regularizer, NumpyLagDependentL1Regularizer
from ...core.constraints_config import MinAbsSumRule, ProcessedConstraintSpec
from ...core.exceptions import ConstraintConfigurationError


class NumpyObjectLoader:
	"""Resolve scikit/numpy callback specs and validate optimizer compatibility."""

	def __init__(self, loading_verbose: bool = False) -> None:
		self._loading_verbose = bool(loading_verbose)

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

	def resolve_regularizer(self, raw_regularizer: Any) -> Any:
		if raw_regularizer is None:
			return None

		if isinstance(raw_regularizer, (NumpyL1Regularizer, NumpyLagDependentL1Regularizer)):
			self._log("regularizer(instance)", raw_regularizer)
			return raw_regularizer

		if not isinstance(raw_regularizer, Mapping):
			return raw_regularizer

		type_name, params = self._extract_typed_spec(raw_regularizer, context="regularizer")
		if type_name in {"l1", "numpy_l1"}:
			resolved = NumpyL1Regularizer(l1=float(params.get("l1", 0.01)))
			self._log("regularizer", resolved)
			return resolved
		if type_name in {"lag_dependent_l1", "lagdependentl1", "numpy_lag_dependent_l1"}:
			resolved = NumpyLagDependentL1Regularizer(
				l1=float(params.get("l1", 0.01)),
				lag_weights=params.get("lag_weights", None),
				max_lags_per_pred=params.get("max_lags_per_pred", None),
				col_offsets=params.get("col_offsets", None),
			)
			self._log("regularizer", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported regularizer type for scikit backend. Supported: l1, lag_dependent_l1"
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
			out_idx = int(rule.get("output_index"))
			feature_indices = tuple(int(v) for v in rule.get("feature_indices", []))
			min_abs_sum = float(rule.get("min_abs_sum", 0.0))
			rules.append(
				MinAbsSumRule(
					output_index=out_idx,
					feature_indices=feature_indices,
					min_abs_sum=min_abs_sum,
				)
			)

		return ProcessedConstraintSpec(mask=mask, rules=tuple(rules))

	def resolve_constraint(self, raw_constraint: Any) -> Any:
		if raw_constraint is None:
			return None

		if isinstance(raw_constraint, (NumpyMaskConstraint, NumpyMaskAndMinAbsSumConstraint)):
			self._log("constraint(instance)", raw_constraint)
			return raw_constraint

		if not isinstance(raw_constraint, Mapping):
			return raw_constraint

		type_name, params = self._extract_typed_spec(raw_constraint, context="constraint")
		if type_name in {"mask", "mask_constraint", "numpy_mask"}:
			if "mask" not in params:
				raise ConstraintConfigurationError("mask constraint requires 'mask'")
			resolved = NumpyMaskConstraint(mask=params["mask"])
			self._log("constraint", resolved)
			return resolved

		if type_name in {"mask_and_min_abs_sum", "mask_min_abs_sum", "numpy_mask_and_min_abs_sum"}:
			spec = self._build_processed_constraint_spec(params)
			resolved = NumpyMaskAndMinAbsSumConstraint(spec=spec, eps=float(params.get("eps", 1e-8)))
			self._log("constraint", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported constraint type for scikit backend. Supported: mask, mask_and_min_abs_sum"
		)

	def resolve_optimizer(self, raw_optimizer: Any) -> None:
		if raw_optimizer is None:
			return None

		raise ConstraintConfigurationError(
			"Scikit backend does not support optimizer objects. Use learning_rate/max_iter/tol in model_config."
		)
