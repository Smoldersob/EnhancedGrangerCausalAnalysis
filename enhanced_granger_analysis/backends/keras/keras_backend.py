from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

from ..base_backend import BackendStrategy
from ...core.components.constraint_spec import PreparedConstraintData
from ...core.components.regularizer_spec import PreparedRegularizerData
from ...core.exceptions import BackendNotAvailableError
from ...core.models.protocol import GrangerModelProtocol


class KerasBackendStrategy(BackendStrategy):
	"""Construct ready Keras components and model adapters."""

	name = "keras"

	_component_names = frozenset({
		"optimizer",
		"loss",
		"callbacks",
		"regularizer",
		"constraint",
	})

	def is_available(self) -> bool:
		try:
			if importlib.util.find_spec("keras") is None:
				return False

			import keras  # noqa: F401

		except ImportError:
			return False

		return True

	def _require_keras(self) -> Any:
		try:
			import keras
		except ImportError as exc:
			raise BackendNotAvailableError(
				"Keras or its configured execution backend is unavailable."
			) from exc

		return keras

	def build_optimizer(
		self,
		spec: Any = None,
		**context: Any,
	) -> Any:
		if context:
			raise TypeError(
				"Keras optimizer construction does not accept context: "
				+ ", ".join(sorted(context))
			)

		self._require_keras()

		from .optimizers.resolver import resolve_optimizer

		optimizer = resolve_optimizer(spec)
		self._log_component("optimizer", optimizer)
		return optimizer

	def build_loss(self, spec: Any = None) -> Any:
		keras = self._require_keras()

		if spec is None:
			loss = keras.losses.MeanSquaredError()
		else:
			from .losses.resolver import resolve_loss

			loss = resolve_loss(spec)

		if loss is None or isinstance(loss, type) or not callable(loss):
			raise TypeError(
				"Loss resolver must return a loss instance or function."
			)

		self._log_component("loss", loss)
		return loss

	def build_callbacks(self, specs: Any = None) -> list[Any]:
		if specs is None:
			return []

		self._require_keras()

		from .callbacks.resolver import resolve_callbacks

		callbacks = resolve_callbacks(specs)
		result = [] if callbacks is None else list(callbacks)

		self._log_component("callbacks", result)
		return result

	def build_regularizer(
		self,
		spec: Any = None,
		*,
		data: PreparedRegularizerData | None = None,
	) -> Any:
		if spec is None:
			return None

		keras = self._require_keras()

		if isinstance(spec, keras.regularizers.Regularizer):
			self._log_component("regularizer", spec)
			return spec

		from .regularizers import resolve_regularizer_spec

		resolved = resolve_regularizer_spec(spec)

		if resolved is None:
			return None

		regularizer_class, _ = resolved

		if data is None:
			raise ValueError(
				"Creating a project regularizer requires "
				"PreparedRegularizerData."
			)

		regularizer = regularizer_class(data=data)

		if not isinstance(regularizer, keras.regularizers.Regularizer):
			raise TypeError(
				"Regularizer construction returned an invalid object."
			)

		self._log_component("regularizer", regularizer)
		return regularizer

	def build_constraint(
		self,
		spec: Any = None,
		*,
		data: PreparedConstraintData | None = None,
	) -> Any:
		if spec is None:
			return None

		keras = self._require_keras()

		if isinstance(spec, keras.constraints.Constraint):
			self._log_component("constraint", spec)
			return spec

		from .constraints import resolve_constraint_spec

		constraint_class, params = resolve_constraint_spec(spec)

		if data is None:
			raise ValueError(
				"Creating a project constraint requires "
				"PreparedConstraintData."
			)

		constraint = constraint_class(
			data=data,
			**params,
		)

		if not isinstance(constraint, keras.constraints.Constraint):
			raise TypeError(
				"Constraint construction returned an invalid object."
			)

		self._log_component("constraint", constraint)
		return constraint
	
	def build_model(
		self,
		spec: Any = None,
		*,
		components: Mapping[str, Any] | None = None,
		config: Mapping[str, Any] | None = None,
	) -> GrangerModelProtocol:
	
		from .models.registry import (
			resolve_model_spec,
			validate_model_instance,
		)
		
		return self._build_registered_model(
			spec,
			resolve_spec=resolve_model_spec,
			validate_instance=validate_model_instance,
			components=components,
			config=config,
			supported_components=self._component_names,
		)