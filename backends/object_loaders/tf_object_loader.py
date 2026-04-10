from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..constraints.tensorflow_constraints import (
	TensorFlowMaskAndMinAbsSumConstraint,
	TensorFlowMaskConstraint,
)
from ..regularizers.tensorflow_regularizers import KerasL1Regularizer, KerasLagDependentL1Regularizer
from ...core.constraints_config import MinAbsSumRule, ProcessedConstraintSpec
from ...core.exceptions import ConstraintConfigurationError


class TensorFlowObjectLoader:
	"""Resolve TensorFlow callback and optimizer specs into Keras objects."""

	def __init__(self, tf_module: Any, loading_verbose: bool = False) -> None:
		self._tf = tf_module
		self._loading_verbose = bool(loading_verbose)

	def set_loading_verbose(self, value: bool) -> None:
		self._loading_verbose = bool(value)

	def _log(self, label: str, value: Any) -> None:
		if self._loading_verbose:
			print(f"[BackendLoader] TensorFlowObjectLoader: {label} -> {value!r}")

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

	def resolve_callback(self, raw_callback: Any) -> Any:
		keras_callback_base = self._tf.keras.callbacks.Callback
		if isinstance(raw_callback, keras_callback_base):
			self._log("callback(instance)", raw_callback)
			return raw_callback

		type_name, params = self._extract_typed_spec(raw_callback, context="callback")

		if type_name in {"early_stopping", "keras_early_stopping", "earlystopping"}:
			resolved = self._tf.keras.callbacks.EarlyStopping(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {
			"reduce_lr_on_plateau",
			"reduce_learning_rate",
			"reduce_lr",
			"reducelronplateau",
			"keras_reduce_lr",
		}:
			resolved = self._tf.keras.callbacks.ReduceLROnPlateau(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"tensorboard", "keras_tensorboard"}:
			resolved = self._tf.keras.callbacks.TensorBoard(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"model_checkpoint", "checkpoint", "keras_checkpoint"}:
			resolved = self._tf.keras.callbacks.ModelCheckpoint(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"csv_logger", "csvlogger", "keras_csv_logger"}:
			resolved = self._tf.keras.callbacks.CSVLogger(**params)
			self._log("callback", resolved)
			return resolved

		if type_name in {"terminate_on_nan", "keras_terminate_on_nan"}:
			resolved = self._tf.keras.callbacks.TerminateOnNaN(**params)
			self._log("callback", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported TensorFlow callback type. Supported: early_stopping, reduce_lr_on_plateau, "
			"tensorboard, model_checkpoint, csv_logger, terminate_on_nan"
		)

	def resolve_callbacks(self, callbacks_cfg: Optional[Sequence[Any]]) -> Optional[List[Any]]:
		if callbacks_cfg is None:
			return None

		if not isinstance(callbacks_cfg, Sequence) or isinstance(callbacks_cfg, (str, bytes)):
			callbacks_cfg = [callbacks_cfg]  # type: ignore[assignment]

		resolved: List[Any] = []
		for raw_cb in callbacks_cfg:
			resolved.append(self.resolve_callback(raw_cb))

		return resolved if resolved else None

	def resolve_optimizer(self, raw_optimizer: Any) -> Any:
		if raw_optimizer is None:
			resolved = self._tf.keras.optimizers.Adam()
			self._log("optimizer", resolved)
			return resolved

		keras_optimizer = self._tf.keras.optimizers.Optimizer
		if isinstance(raw_optimizer, keras_optimizer):
			self._log("optimizer(instance)", raw_optimizer)
			return raw_optimizer

		if isinstance(raw_optimizer, str):
			resolved = self._tf.keras.optimizers.get(raw_optimizer)
			self._log("optimizer", resolved)
			return resolved

		if isinstance(raw_optimizer, Mapping):
			# Support both Keras-native format:
			#   {"class_name": "Adam", "config": {...}}
			# and compact config format used in this project:
			#   {"type": "adam", "learning_rate": 0.001, ...}
			if "class_name" in raw_optimizer:
				resolved = self._tf.keras.optimizers.get(raw_optimizer)
				self._log("optimizer", resolved)
				return resolved

			type_name, params = self._extract_typed_spec(raw_optimizer, context="optimizer")
			keras_name_map = {
				"adam": "Adam",
				"sgd": "SGD",
				"rmsprop": "RMSprop",
				"adagrad": "Adagrad",
				"adamax": "Adamax",
				"nadam": "Nadam",
				"ftrl": "Ftrl",
			}
			class_name = keras_name_map.get(type_name, type_name)
			resolved = self._tf.keras.optimizers.get(
				{
					"class_name": class_name,
					"config": params,
				}
			)
			self._log("optimizer", resolved)
			return resolved

		if isinstance(raw_optimizer, type) and issubclass(raw_optimizer, keras_optimizer):
			resolved = raw_optimizer()
			self._log("optimizer", resolved)
			return resolved

		if callable(raw_optimizer):
			candidate = raw_optimizer()
			if isinstance(candidate, keras_optimizer):
				self._log("optimizer(callable)", candidate)
				return candidate
			raise ConstraintConfigurationError(
				"TensorFlow optimizer callable must return tf.keras.optimizers.Optimizer"
			)

		raise ConstraintConfigurationError(
			"Unsupported TensorFlow optimizer spec. Use string/object spec, optimizer instance, class, or callable."
		)

	def resolve_regularizer(self, raw_regularizer: Any) -> Any:
		if raw_regularizer is None:
			return None

		keras_regularizer = self._tf.keras.regularizers.Regularizer
		if isinstance(raw_regularizer, keras_regularizer):
			self._log("regularizer(instance)", raw_regularizer)
			return raw_regularizer

		if not isinstance(raw_regularizer, Mapping):
			return raw_regularizer

		type_name, params = self._extract_typed_spec(raw_regularizer, context="regularizer")
		if type_name in {"l1", "keras_l1", "tensorflow_l1"}:
			resolved = KerasL1Regularizer(l1=float(params.get("l1", 0.01)))
			self._log("regularizer", resolved)
			return resolved
		if type_name in {"lag_dependent_l1", "lagdependentl1", "keras_lag_dependent_l1", "tensorflow_lag_dependent_l1"}:
			resolved = KerasLagDependentL1Regularizer(
				l1=float(params.get("l1", 0.01)),
				lag_weights=params.get("lag_weights", None),
				max_lags_per_pred=params.get("max_lags_per_pred", None),
				col_offsets=params.get("col_offsets", None),
			)
			self._log("regularizer", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported TensorFlow regularizer type. Supported: l1, lag_dependent_l1"
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

		keras_constraint = self._tf.keras.constraints.Constraint
		if isinstance(raw_constraint, keras_constraint):
			self._log("constraint(instance)", raw_constraint)
			return raw_constraint

		if not isinstance(raw_constraint, Mapping):
			return raw_constraint

		type_name, params = self._extract_typed_spec(raw_constraint, context="constraint")
		if type_name in {"mask", "mask_constraint", "keras_mask", "tensorflow_mask"}:
			if "mask" not in params:
				raise ConstraintConfigurationError("mask constraint requires 'mask'")
			resolved = TensorFlowMaskConstraint(mask=params["mask"])
			self._log("constraint", resolved)
			return resolved

		if type_name in {"mask_and_min_abs_sum", "mask_min_abs_sum", "keras_mask_and_min_abs_sum", "tensorflow_mask_and_min_abs_sum"}:
			spec = self._build_processed_constraint_spec(params)
			resolved = TensorFlowMaskAndMinAbsSumConstraint(spec=spec, eps=float(params.get("eps", 1e-8)))
			self._log("constraint", resolved)
			return resolved

		raise ConstraintConfigurationError(
			"Unsupported TensorFlow constraint type. Supported: mask, mask_and_min_abs_sum"
		)
