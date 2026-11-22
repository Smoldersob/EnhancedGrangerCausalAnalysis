from collections.abc import Mapping
from dataclasses import dataclass, field

from ...core.lags.specification import Aggregation


@dataclass(frozen=True)
class LagEngineConfig:
    """Execution settings for lag fitting and preparation."""

    max_lag: int = 12
    use_lag_zero: bool = False
    aggregation: Aggregation = "union"
    selection_frequency: float = 0.5
    predictor_overrides: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    pair_overrides: Mapping[str, Mapping[str, tuple[int, ...]]] = field(
        default_factory=dict
    )
