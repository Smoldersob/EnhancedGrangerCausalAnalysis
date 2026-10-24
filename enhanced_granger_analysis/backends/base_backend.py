from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Callable
from typing import Any

from ..core.components.constraint_spec import PreparedConstraintData
from ..core.components.regularizer_spec import PreparedRegularizerData
from ..core.models.protocol import GrangerModelProtocol


class BackendStrategy(ABC):
	"""Factory interface for backend components and model adapters.

	Component builders accept specifications or ready instances.

	build_model() accepts ready components only. It does not resolve
	optimizer, loss, callback, regularizer or constraint specifications.
	"""

	name: str

	def __init__(self, *, loading_verbose: bool = False) -> None:
		self._loading_verbose = bool(loading_verbose)

	def set_loading_verbose(self, value: bool) -> None:
		self._loading_verbose = bool(value)

	def _log_component(self, label: str, value: Any) -> None:
		if self._loading_verbose:
			print(f"[Backend:{self.name}] {label} -> {value!r}")

	@abstractmethod
	def is_available(self) -> bool:
		"""Return whether the backend can be used."""
		raise NotImplementedError

	@abstractmethod
	def build_optimizer(
		self,
		spec: Any = None,
		**context: Any,
	) -> Any:
		"""Return a ready optimizer.

		Context may contain backend-specific construction dependencies,
		for example trainable parameters.
		"""
		raise NotImplementedError

	@abstractmethod
	def build_loss(self, spec: Any = None) -> Any:
		"""Return a ready loss object or function."""
		raise NotImplementedError

	@abstractmethod
	def build_callbacks(self, specs: Any = None) -> list[Any]:
		"""Return ready callbacks; None means an empty list."""
		raise NotImplementedError

	@abstractmethod
	def build_regularizer(
		self,
		spec: Any = None,
		*,
		data: PreparedRegularizerData | None = None,
	) -> Any:
		"""Return a ready regularizer; None disables regularization."""
		raise NotImplementedError

	@abstractmethod
	def build_constraint(
		self,
		spec: Any = None,
		*,
		data: PreparedConstraintData | None = None,
	) -> Any:
		"""Return a ready constraint; None disables this component."""
		raise NotImplementedError

	@abstractmethod
	def build_model(
		self,
		spec: Any = None,
		*,
		components: Mapping[str, Any] | None = None,
		config: Mapping[str, Any] | None = None,
	) -> GrangerModelProtocol:
		"""Return a model adapter without initializing or training it.

		spec:
			None, registered name, specification, class or ready model.

		components:
			Already prepared backend components.

		config:
			Non-component constructor parameters.

		A ready model must not be combined with constructor overrides.
		"""
		raise NotImplementedError

	def _build_registered_model(
		self,
		spec: Any,
		*,
		resolve_spec: Callable[..., Any],
		validate_instance: Callable[..., GrangerModelProtocol],
		components: Mapping[str, Any] | None = None,
		config: Mapping[str, Any] | None = None,
		supported_components: frozenset[str],
		ignored_components: frozenset[str] = frozenset(),
	) -> GrangerModelProtocol:
		"""Construct a model using an already selected backend registry."""
		target, params = resolve_spec(
			"conditional_linear" if spec is None else spec
		)

		if target is None:
			raise ValueError(f"No {self.name} model was selected.")

		if not isinstance(target, type):
			if components or config:
				raise ValueError(
					"A ready model cannot be combined with "
					"components or constructor parameters."
				)
			return target

		component_values = (
			{} if components is None else dict(components)
		)
		config_values = (
			{} if config is None else dict(config)
		)

		for values in (params, config_values, component_values):
			if any(not isinstance(key, str) for key in values):
				raise TypeError("Model argument names must be strings.")

		known_components = supported_components | ignored_components

		unknown = set(component_values) - known_components
		if unknown:
			raise ValueError(
				"Unsupported component names: "
				+ ", ".join(sorted(unknown))
			)

		misplaced = (
			set(params) | set(config_values)
		) & known_components
		if misplaced:
			raise ValueError(
				"Pass ready components through 'components': "
				+ ", ".join(sorted(misplaced))
			)

		for name in ignored_components:
			value = component_values.pop(name, None)
			if value is not None:
				raise ValueError(
					f"Backend {self.name!r} does not accept "
					f"a non-None {name!r} component."
				)

		duplicated = (
			(set(params) & set(config_values))
			| (set(params) & set(component_values))
			| (set(config_values) & set(component_values))
		)
		if duplicated:
			raise ValueError(
				"Model arguments were supplied more than once: "
				+ ", ".join(sorted(duplicated))
			)

		model = target(
			**params,
			**config_values,
			**component_values,
		)

		return validate_instance(model)