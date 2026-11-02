from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.exceptions import DataValidationError, RegistryError
from ..core.initialization import InitializationContext
from ..core.models.protocol import GrangerModelProtocol
from ..initializers.registry import resolve_initializer_spec
from ..preprocessing.prepare_data import PreparedData


class ModelInitializer:
	"""Compute initial weights without assigning them to the model."""

	def compute_weights(
		self,
		*,
		model: GrangerModelProtocol,
		prepared: PreparedData,
		initializer_spec: Any = None,
		row_indices: np.ndarray | None = None,
		initialization_mask: np.ndarray | None = None,
	) -> Sequence[np.ndarray] | None:
		"""Compute weights from selected observations and an optional mask."""
		try:
			initializer_class, params = resolve_initializer_spec(
				initializer_spec
			)
		except (RegistryError, TypeError, ValueError) as exc:
			raise DataValidationError(
				f"Invalid initializer specification: {exc}"
			) from exc

		if initializer_class is None:
			return None

		X = prepared.X_scaled
		y = prepared.y_scaled

		if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
			raise DataValidationError(
				"Initialization requires aligned 2D input and target arrays."
			)

		if row_indices is not None:
			indices = np.asarray(row_indices)

			if (
				indices.ndim != 1
				or indices.size == 0
				or indices.dtype.kind not in {"i", "u"}
			):
				raise DataValidationError(
					"Initialization indices must be a non-empty "
					"1D integer array."
				)

			if (
				np.any(indices < 0)
				or np.any(indices >= len(X))
				or np.any(indices[1:] <= indices[:-1])
			):
				raise DataValidationError(
					"Initialization indices must be valid, unique "
					"and increasing."
				)

			X = X[indices]
			y = y[indices]

		mask = (
			prepared.base_mask
			if initialization_mask is None
			else initialization_mask
		)

		initializer = initializer_class(**params)

		context = InitializationContext(
			X=np.array(X, copy=True),
			y=np.array(y, copy=True),
			weights=tuple(
				np.asarray(weight).copy()
				for weight in model.get_weights()
			),
			mask=(
				None
				if mask is None
				else np.asarray(mask).copy()
			),
		)

		return initializer(context)