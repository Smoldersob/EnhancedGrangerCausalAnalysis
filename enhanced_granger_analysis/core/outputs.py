from dataclasses import dataclass

import pandas as pd
import pandas as pd
from ..results.causality_matrix import CausalityMatrix

@dataclass
class SimpleGrangerOutput:
	"""Output container returned by SimpleGrangerAPI.fit()."""

	causality_matrix: CausalityMatrix
	p_value: pd.DataFrame
	sign: pd.DataFrame


__all__ = ["SimpleGrangerOutput"]
