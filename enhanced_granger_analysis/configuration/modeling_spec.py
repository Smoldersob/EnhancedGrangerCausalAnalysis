from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.components.constraint_spec import RelationConstraintsSpec


def _default_constraint_spec() -> dict[str, Any]:
	"""Return an independent default constraint declaration."""

	return {
		"type": "mask_and_min_abs_sum",
		"params": {},
	}


@dataclass(frozen=True)
class ModelingSpec:
	"""Describe model construction without resolving components.

	Component declarations are passed to existing parsers, backend
	strategies, and resolvers by the execution services.
	"""

	backend: str | None = None

	model_config: Mapping[str, Any] = field(
		default_factory=dict
	)

	constraint_spec: Any = field(
		default_factory=_default_constraint_spec
	)
	relations: Any = field(
		default_factory=RelationConstraintsSpec
	)

	regularizer_spec: Any = None
	callbacks: Any = None
	initializer: Any = None