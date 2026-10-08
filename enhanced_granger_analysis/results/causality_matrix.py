from __future__ import annotations

from dataclasses import dataclass,	field
from typing import Any
import pandas as pd

from ..core.exceptions import CausalityMatrixError

@dataclass
class CausalityMatrix:
	"""Wrapper around a 2D DataFrame with fixed effects x causes indexing."""

	data: pd.DataFrame
	name: str | None = None
	metadata: dict[str, Any] = field(default_factory=dict)

	def threshold(self, threshold: float = 0.01) -> pd.DataFrame:
		try:
			return (self.data < threshold).astype(int)
		except Exception as e:
			raise CausalityMatrixError(f"Error occurred while applying threshold: {e}")