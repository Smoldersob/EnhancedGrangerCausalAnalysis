import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

from .lag import LagEngine, LagConfiguration, BaseLagSelector
from .stationarity import StationarityTransformer
from .missing import BaseImputer
from .sampling import NoSampling, SamplingStrategy

_DEFAULT_STATIONARITY = object()

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
	stationarity_transformer: Any = _DEFAULT_STATIONARITY,
	lag_engine: Optional[LagEngine] = None
	x_scaler: Any = None
	y_scaler: Any = None
	X_scaled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	y_scaled: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	X_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	y_raw: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
	base_mask: Optional[np.ndarray] = None
	requested_roles: VariableRoles = field(
		default_factory=lambda: VariableRoles((), (), ())
	)
	effective_roles: VariableRoles = field(
		default_factory=lambda: VariableRoles((), (), ())
	)
	cause_slices: dict[str, slice] = field(default_factory=dict)
	removed_variables: dict[str, RemovedVariable] = field(default_factory=dict)
	sample_indices: np.ndarray = field(
		default_factory=lambda: np.empty(0, dtype=int)
	)

	@property
	def causes(self) -> tuple[str, ...]:
		return self.effective_roles.causes

	@property
	def effects(self) -> tuple[str, ...]:
		return self.effective_roles.effects

	@property
	def tested_causes(self) -> tuple[str, ...]:
		return self.effective_roles.tested_causes

	@property
	def tested_cause_slices(self) -> dict[str, slice]:
		return {
			name: self.cause_slices[name]
			for name in self.effective_roles.tested_causes
		}

	@property
	def predictor_slices(self) -> dict[str, slice]:
		"""Compatibility alias; prefer cause_slices."""
		return dict(self.cause_slices)

	@classmethod
	def empty(cls) -> "PreparedData":
		"""Create an empty stage container that can be filled progressively."""
		return cls(
			stationarity_transformer=_DEFAULT_STATIONARITY,
			lag_engine=None,
			x_scaler=None,
			y_scaler=None,
			X_scaled=np.empty((0, 0), dtype=float),
			y_scaled=np.empty((0, 0), dtype=float),
			X_raw=np.empty((0, 0), dtype=float),
			y_raw=np.empty((0, 0), dtype=float),
			base_mask=None,
			requested_roles=VariableRoles((), (), ()),
			effective_roles=VariableRoles((), (), ()),
			cause_slices={},
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
		schema_policy: str = "union",
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
		random_state: Optional[int] = None,
		sampling_strategy: Optional[SamplingStrategy] = NoSampling(),
		x_scaler: Optional[Any] = "standard",
		y_scaler: Optional[Any] = "standard",
) -> PreparedData:
	from .pipeline import DataPreparer, DataPreparationConfig

	if stationarity_transformer is _DEFAULT_STATIONARITY:
		stationarity_transformer = StationarityTransformer()

	if sampling_strategy is None:
		sampling_strategy = NoSampling()

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