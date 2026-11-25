"""Deterministic sample-selection strategies for prepared time-series data."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..core.data.prepared import PreparedData
from ..core.data.split import DataSplit
from ..core.exceptions import DataValidationError

class SamplingStrategy(ABC):
	"""Select row indices without changing their source ordering."""

	@abstractmethod
	def select(
		self,
		n_samples: int,
		*,
		n_keep: int,
		random_state: np.random.Generator,
	) -> NDArray[np.int_]:
		"""Return selected row indices in the order used by the strategy."""


class NoSampling(SamplingStrategy):
	"""Keep all rows in their original order."""

	def select(
		self,
		n_samples: int,
		*,
		n_keep: int,
		random_state: np.random.Generator,
	) -> NDArray[np.int_]:
		return np.arange(n_samples, dtype=int)


class StridedSampling(SamplingStrategy):
	"""Keep evenly spaced rows while preserving chronological order."""

	def select(
		self,
		n_samples: int,
		*,
		n_keep: int,
		random_state: np.random.Generator,
	) -> NDArray[np.int_]:
		if n_keep >= n_samples:
			return np.arange(n_samples, dtype=int)
		return np.linspace(0, n_samples - 1, n_keep, dtype=int)


class RandomSampling(SamplingStrategy):
	"""Select distinct rows reproducibly using the supplied generator."""

	def select(
		self,
		n_samples: int,
		*,
		n_keep: int,
		random_state: np.random.Generator,
	) -> NDArray[np.int_]:
		if n_keep >= n_samples:
			return np.arange(n_samples, dtype=int)
		return np.sort(random_state.choice(n_samples, size=n_keep, replace=False))


def sample_prepared_data(
	prepared: PreparedData,
	*,
	fraction: float = 1.0,
	max_samples: int | None = None,
	random_state: int | None = None,
	strategy: SamplingStrategy | None = None,
) -> PreparedData:
	"""Return a sampled result without changing its fitted representation."""
	if not isinstance(prepared, PreparedData):
		raise TypeError("prepared must be PreparedData.")
	if isinstance(fraction, (bool, np.bool_)) or not isinstance(
		fraction, (int, float, np.number)
	) or not 0 < fraction <= 1:
		raise DataValidationError("fraction must be in (0, 1].")
	if max_samples is not None and (
		isinstance(max_samples, (bool, np.bool_))
		or not isinstance(max_samples, (int, np.integer))
		or max_samples <= 0
	):
		raise DataValidationError("max_samples must be a positive integer or None.")
	if random_state is not None and (
		isinstance(random_state, (bool, np.bool_))
		or not isinstance(random_state, (int, np.integer))
		or random_state < 0
	):
		raise DataValidationError("random_state must be a non-negative integer or None.")
	selected_strategy = NoSampling() if strategy is None else strategy
	if not isinstance(selected_strategy, SamplingStrategy):
		raise TypeError("strategy must be a SamplingStrategy or None.")

	n_rows = prepared.X_raw.shape[0]
	n_keep = max(1, int(np.floor(fraction * n_rows)))
	if max_samples is not None:
		n_keep = min(n_keep, int(max_samples))
	if n_keep < n_rows and (
		len(prepared.split.part_names) != 1
		or prepared.split.part_names[0] != "training"
		or prepared.split.indices_for("training").size != n_rows
	):
		raise DataValidationError(
			"Sampling cannot remove rows while validation or test assignments are active."
		)
	indices = np.asarray(selected_strategy.select(
		n_rows,
		n_keep=n_keep,
		random_state=np.random.default_rng(random_state),
	))
	if (
		indices.ndim != 1
		or indices.dtype.kind not in {"i", "u"}
		or indices.size == 0
		or np.any(indices >= n_rows)
		or (indices.dtype.kind == "i" and np.any(indices < 0))
		or np.any(indices[1:] <= indices[:-1])
	):
		raise DataValidationError(
			"Sampling strategy must return valid, unique, increasing row indices."
		)
	indices = indices.astype(np.intp, copy=False)
	split = (
		prepared.split
		if indices.size == n_rows
		else DataSplit.all_training(indices.size)
	)
	return PreparedData(
		X_raw=prepared.X_raw[indices],
		y_raw=prepared.y_raw[indices],
		X_scaled=prepared.X_scaled[indices],
		y_scaled=prepared.y_scaled[indices],
		feature_layout=prepared.feature_layout,
		requested_roles=prepared.requested_roles,
		effective_roles=prepared.effective_roles,
		base_mask=prepared.base_mask,
		split=split,
		sample_indices=prepared.sample_indices[indices],
		target_inverse=prepared.target_inverse,
		segments=prepared.segments,
		variable_descriptions=prepared.variable_descriptions,
		removed_variables=prepared.removed_variables,
	)


__all__ = [
	"SamplingStrategy",
	"NoSampling",
	"StridedSampling",
	"RandomSampling",
	"sample_prepared_data",
]
