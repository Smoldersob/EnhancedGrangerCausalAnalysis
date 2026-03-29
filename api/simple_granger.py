from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import BackendNotAvailableError, DataValidationError
from ..core.outputs import SimpleGrangerOutput
from ..preprocessing.stationarity import StationarityTransformer
from ..results.causality_matrix import CausalityMatrix


def _resolve_lag_order(data: pd.DataFrame, lag: Optional[int], lag_max: int) -> int:
	if lag is not None:
		if lag <= 0:
			raise ValueError("lag must be > 0")
		return int(lag)

	try:
		from statsmodels.tsa.vector_ar.var_model import VAR
	except Exception as exc:  # pragma: no cover - optional dependency
		raise BackendNotAvailableError("statsmodels is required for simple pair-wise API") from exc

	fit = VAR(data).fit(maxlags=int(lag_max), ic="aic")
	selected = int(getattr(fit, "k_ar", 0))
	return max(selected, 1)


class SimpleGrangerAPI:
	"""
	Pair-wise statsmodels-based Granger causality analysis.

	The output is represented by CausalityMatrix plus auxiliary p_value and sign tables.
	Using GrangerAnalysisResults here would be artificial because statsmodels already
	returns test statistics directly (without base/reference model snapshots).
	"""

	def fit(
		self,
		data: pd.DataFrame,
		causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		test: str = "ssr_chi2test",
		lag: Optional[int] = None,
		lag_max: int = 20,
		stationarity_transformer: Optional[StationarityTransformer] = None,
		threshold: float = 0.01,
	) -> SimpleGrangerOutput:
		if not isinstance(data, pd.DataFrame):
			raise DataValidationError("data must be a pandas DataFrame")

		columns = list(data.columns)
		causes_list = list(causes) if causes is not None else columns
		effects_list = list(effects) if effects is not None else columns

		unknown_causes = [c for c in causes_list if c not in columns]
		unknown_effects = [e for e in effects_list if e not in columns]
		if unknown_causes or unknown_effects:
			raise DataValidationError(
				f"Unknown variables detected. causes={unknown_causes}, effects={unknown_effects}"
			)

		transformer = stationarity_transformer or StationarityTransformer()
		stationary = transformer.fit_transform([data])[0]

		lag_order = _resolve_lag_order(stationary[columns], lag=lag, lag_max=lag_max)

		try:
			from statsmodels.tsa.stattools import grangercausalitytests
		except Exception as exc:  # pragma: no cover - optional dependency
			raise BackendNotAvailableError("statsmodels is required for simple pair-wise API") from exc

		p_value_df = pd.DataFrame(
			np.ones((len(effects_list), len(causes_list)), dtype=np.float64),
			index=effects_list,
			columns=causes_list,
		)
		sign_df = pd.DataFrame(
			np.ones((len(effects_list), len(causes_list)), dtype=np.float64),
			index=effects_list,
			columns=causes_list,
		)

		for cause_name in causes_list:
			for effect_name in effects_list:
				test_result = grangercausalitytests(
					stationary[[effect_name, cause_name]],
					maxlag=[lag_order],
					verbose=False,
				)
				p_raw = float(test_result[lag_order][0][test][1])
				p_value_df.loc[effect_name, cause_name] = p_raw

				# Coefficients associated with lagged cause terms in this 2-variable setup.
				params = np.asarray(test_result[lag_order][1][1].params, dtype=np.float64)
				coef_slice = params[lag_order:-1]
				if coef_slice.size == 0:
					sign_val = 0.0
				else:
					idx = int(np.argmax(np.abs(coef_slice)))
					sign_val = float(np.sign(coef_slice[idx]))
				sign_df.loc[effect_name, cause_name] = sign_val

		binary_df = (p_value_df < threshold).astype(np.float64)
		signed_binary = binary_df.copy()
		signed_binary[binary_df == 1.0] = signed_binary[binary_df == 1.0] * sign_df[binary_df == 1.0]

		return SimpleGrangerOutput(
			causality_matrix=CausalityMatrix(data=signed_binary),
			p_value=p_value_df,
			sign=sign_df,
		)


__all__ = ["SimpleGrangerAPI", "SimpleGrangerOutput"]
