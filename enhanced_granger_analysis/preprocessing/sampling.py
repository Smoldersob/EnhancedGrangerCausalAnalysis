"""Deterministic sample-selection strategies for prepared time-series data."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


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


__all__ = ["SamplingStrategy", "NoSampling", "StridedSampling", "RandomSampling"]
