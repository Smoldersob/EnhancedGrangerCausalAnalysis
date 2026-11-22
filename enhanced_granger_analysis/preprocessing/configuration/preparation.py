from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ...preprocessing.missing import BaseImputer
from ...preprocessing.sampling import NoSampling, SamplingStrategy


@dataclass(frozen=True)
class DataPreparationConfig:
    """Configuration shared by staged preprocessing."""

    schema_policy: str = "error"
    backend_sample_fraction: float = 1.0
    backend_max_samples: Optional[int] = None
    random_state: Optional[int] = None
    x_scaler: Any = "standard"
    y_scaler: Any = "standard"
    lag_config: Any = None
    lag_selector: Any = None
    lag_selector_params: Mapping[str, Any] = field(default_factory=dict)
    lag_n_jobs: int = 1
    stationarity: Any = None
    missing_value_imputer: Optional[BaseImputer] = None
    sampling_strategy: Optional[SamplingStrategy] = field(default_factory=NoSampling)
    splitter: Any = None
