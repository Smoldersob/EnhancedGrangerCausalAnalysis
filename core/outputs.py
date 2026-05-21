from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from ..preprocessing.lag.lag_engine import LagEngine
from ..preprocessing.stationarity import StationarityTransformer
from ..results.causality_matrix import CausalityMatrix
from ..results.granger_results import GrangerAnalysisResults


@dataclass
class MultitaskGrangerOutput:
	"""Output container returned by MultiTaskGrangerAPI.fit()."""

	results: GrangerAnalysisResults
	base_model: Any
	reference_models: Dict[str, Any]
	stationarity_transformer: StationarityTransformer
	lag_engine: LagEngine
	X_scaler: Any
	y_scaler: Any
	preparation_time_seconds: float = 0.0
	prepared_data: Any = None


@dataclass
class SimpleGrangerOutput:
	"""Output container returned by SimpleGrangerAPI.fit()."""

	causality_matrix: CausalityMatrix
	p_value: pd.DataFrame
	sign: pd.DataFrame


__all__ = ["MultitaskGrangerOutput", "SimpleGrangerOutput"]
