"""Explicit missing-value handling strategies for time-series segments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from ..core.exceptions import DataValidationError


class BaseImputer(ABC):
	"""Template for imputers that operate independently on each segment."""

	def fit(self, data_list: Sequence[pd.DataFrame]) -> "BaseImputer":
		self._validate(data_list)
		return self

	@abstractmethod
	def transform(self, data_list: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
		"""Transform segments without crossing their time boundaries."""

	def fit_transform(
		self, data_list: Sequence[pd.DataFrame]
	) -> list[pd.DataFrame]:
		return self.fit(data_list).transform(data_list)

	@staticmethod
	def _validate(data_list: Sequence[pd.DataFrame]) -> None:
		if not data_list:
			raise DataValidationError("data_list must contain at least one DataFrame")
		if not all(isinstance(data, pd.DataFrame) for data in data_list):
			raise DataValidationError(
				"Every element of data_list must be a pandas DataFrame"
			)


class ConstantImputer(BaseImputer):
	"""Replace missing values with one constant."""

	def __init__(self, value: float = 0.0) -> None:
		self.value = value

	def transform(self, data_list: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
		self._validate(data_list)
		return [data.fillna(self.value) for data in data_list]


class MeanImputer(BaseImputer):
	"""Replace missing values with per-column means fitted across segments."""

	def fit(self, data_list: Sequence[pd.DataFrame]) -> "MeanImputer":
		self._validate(data_list)
		self.statistics_ = pd.concat(data_list, axis=0).mean(axis=0)
		return self

	def transform(self, data_list: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
		self._validate(data_list)
		if not hasattr(self, "statistics_"):
			raise DataValidationError("MeanImputer is not fitted")
		return [data.fillna(self.statistics_) for data in data_list]


class ForwardFillImputer(BaseImputer):
	"""Forward-fill missing values independently inside each segment."""

	def transform(self, data_list: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
		self._validate(data_list)
		return [data.ffill() for data in data_list]


class LinearInterpolationImputer(BaseImputer):
	"""Linearly interpolate missing values independently inside each segment."""

	def __init__(self, *, limit_direction: str = "both") -> None:
		self.limit_direction = limit_direction

	def transform(self, data_list: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
		self._validate(data_list)
		return [
			data.interpolate(
				method="linear", limit_direction=self.limit_direction
			)
			for data in data_list
		]


class DropRowsImputer(BaseImputer):
	"""Drop rows containing at least one missing value per segment."""

	def transform(self, data_list: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
		self._validate(data_list)
		return [data.dropna() for data in data_list]


__all__ = [
		"BaseImputer",
		"ConstantImputer",
		"MeanImputer",
		"ForwardFillImputer",
		"LinearInterpolationImputer",
		"DropRowsImputer",
	]
