import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Sequence

from .lag import LagEngine, LagConfiguration, BaseLagSelector
from .stationarity import StationarityTransformer
from .missing import BaseImputer
from .sampling import NoSampling, SamplingStrategy


@dataclass(frozen=True)
class VariableRoles:
	"""Requested or effective variable roles used by the preparation pipeline."""

	effects: tuple[str, ...]
	causes: tuple[str, ...]
	tested_causes: tuple[str, ...]


@dataclass(frozen=True)
class RemovedVariable:
	"""Description of a variable removed during redundant-column cleanup."""

	name: str
	reason: Literal["constant", "duplicate"]
	removed_roles: tuple[Literal["effect", "cause", "tested_cause"], ...]
	retained_variable: Optional[str] = None


@dataclass
class PreparedData:
	stationarity_transformer: Optional[StationarityTransformer] = None
	lag_engine: Optional[LagEngine] = None
	col_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
	x_scaler: Any = None
	y_scaler: Any = None
	X_scaled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	y_scaled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	X_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	y_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	base_mask: Optional[np.ndarray] = None
	causes: List[str] = field(default_factory=list)
	tested_causes: List[str] = field(default_factory=list)
	effects: List[str] = field(default_factory=list)
	predictor_slices: dict[str, slice] = field(default_factory=dict)
	removed_variable_info: dict[str, dict[str, Any]] = field(default_factory=dict)
	requested_roles: VariableRoles = field(
		default_factory=lambda: VariableRoles((), (), ())
	)
	effective_roles: VariableRoles = field(
		default_factory=lambda: VariableRoles((), (), ())
	)
	cause_slices: dict[str, slice] = field(default_factory=dict)
	tested_cause_slices: dict[str, slice] = field(default_factory=dict)
	removed_variables: dict[str, RemovedVariable] = field(default_factory=dict)
	sample_indices: np.ndarray = field(
		default_factory=lambda: np.empty(0, dtype=int)
	)

	@classmethod
	def empty(cls) -> "PreparedData":
		"""Create an empty stage container that can be filled progressively."""
		return cls(
			stationarity_transformer=None,
			lag_engine=None,
			col_offsets=np.empty(0, dtype=int),
			x_scaler=None,
			y_scaler=None,
			X_scaled=np.empty((0, 0), dtype=float),
			y_scaled=np.empty((0, 0), dtype=float),
			X_raw=np.empty((0, 0), dtype=float),
			y_raw=np.empty((0, 0), dtype=float),
			base_mask=None,
			causes=[],
			tested_causes=[],
			effects=[],
			predictor_slices={},
			removed_variable_info={},
			requested_roles=VariableRoles((), (), ()),
			effective_roles=VariableRoles((), (), ()),
			cause_slices={},
			tested_cause_slices={},
			removed_variables={},
			sample_indices=np.empty(0, dtype=int),
		)


def prepare_data(
		data: pd.DataFrame | Sequence[pd.DataFrame],
		causes: Optional[Sequence[str]] = None,
		tested_causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		lag_config: Optional[LagConfiguration] = None,
		lag_selector: Optional[BaseLagSelector] = None,
		stationarity_transformer: Optional[StationarityTransformer] = None,
		missing_value_imputer: Optional[BaseImputer] = None,
		schema_policy: str = "error",
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
		random_state: Optional[int] = None,
		sampling_strategy: Optional[SamplingStrategy] = NoSampling(),
		x_scaler: Optional[Any] = "standard",
		y_scaler: Optional[Any] = "standard",
) -> PreparedData:
	from .pipeline import DataPreparer, DataPreparationConfig

	config = DataPreparationConfig(
		schema_policy=schema_policy,
		backend_sample_fraction=backend_sample_fraction,
		backend_max_samples=backend_max_samples,
		random_state=random_state,
		x_scaler=x_scaler,
		y_scaler=y_scaler,
		lag_config=lag_config,
		lag_selector=lag_selector,
		stationarity_transformer=stationarity_transformer,
		missing_value_imputer=missing_value_imputer,
		sampling_strategy=sampling_strategy,
	)
	preparer = DataPreparer(config=config)
	roles = VariableRoles(
		causes=tuple(causes) if causes is not None else (),
		tested_causes=tuple(tested_causes) if tested_causes is not None else (),
		effects=tuple(effects) if effects is not None else (),
	)
	return preparer.fit(data, roles=roles).prepared_data_