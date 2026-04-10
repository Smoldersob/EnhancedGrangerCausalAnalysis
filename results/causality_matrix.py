from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.exceptions import CausalityMatrixError


@dataclass
class CausalityMatrix:
	"""Wrapper around a 2D DataFrame with fixed effects x causes indexing."""

	data: pd.DataFrame

	@classmethod
	def zeros(cls, effects: Iterable[str], causes: Iterable[str]) -> "CausalityMatrix":
		effects_list = list(effects)
		causes_list = list(causes)
		df = pd.DataFrame(
			np.zeros((len(effects_list), len(causes_list)), dtype=np.float64),
			index=effects_list,
			columns=causes_list,
		)
		return cls(data=df)

	@classmethod
	def ones(cls, effects: Iterable[str], causes: Iterable[str]) -> "CausalityMatrix":
		effects_list = list(effects)
		causes_list = list(causes)
		df = pd.DataFrame(
			np.ones((len(effects_list), len(causes_list)), dtype=np.float64),
			index=effects_list,
			columns=causes_list,
		)
		return cls(data=df)

	def set_column(self, cause: str, values: NDArray[np.float64]) -> None:
		arr = np.asarray(values, dtype=np.float64)
		if arr.shape[0] != self.data.shape[0]:
			raise CausalityMatrixError(
				f"Column vector length mismatch for cause '{cause}': {arr.shape[0]} vs {self.data.shape[0]}"
			)
		self.data.loc[:, cause] = arr

	def threshold(self, threshold: float) -> pd.DataFrame:
		return (self.data < threshold).astype(int)


@dataclass
class CausalityMatrices:
	"""Bundle of all matrices used in Granger analysis reporting."""

	base_error: CausalityMatrix
	ref_error: CausalityMatrix
	f_test: CausalityMatrix
	p_value: CausalityMatrix
	sign: CausalityMatrix

	@classmethod
	def create(cls, effects: Iterable[str], causes: Iterable[str]) -> "CausalityMatrices":
		return cls(
			base_error=CausalityMatrix.zeros(effects, causes),
			ref_error=CausalityMatrix.zeros(effects, causes),
			f_test=CausalityMatrix.ones(effects, causes),
			p_value=CausalityMatrix.ones(effects, causes),
			sign=CausalityMatrix.ones(effects, causes),
		)

	def result(self, threshold: float = 0.01, with_sign: bool = False) -> pd.DataFrame:
		ans = self.p_value.threshold(threshold)
		if with_sign:
			signed = self.sign.data.copy()
			ans = ans.astype(np.float64)
			ans[ans == 1] = ans[ans == 1] * signed[ans == 1]
			return ans
		return ans
