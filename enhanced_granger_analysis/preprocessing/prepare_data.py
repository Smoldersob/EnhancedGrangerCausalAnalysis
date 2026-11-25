import pandas as pd
from typing import Any, Mapping, Optional, Sequence

from .missing import BaseImputer
from .sampling import NoSampling, SamplingStrategy

from ..core.data.analysis_roles import VariableRoles
from ..core.data.prepared import PreparedData
from ..core.data.variables import ValueType, VariableDescription


def prepare_data(
		data: pd.DataFrame | Sequence[pd.DataFrame],
		causes: Optional[Sequence[str]] = None,
		tested_causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		lag_config: Any = None,
		lag_selector: Any = None,
		lag_selector_params: Optional[Mapping[str, Any]] = None,
		lag_n_jobs: int = 1,
		stationarity: Any = None,
		missing_value_imputer: Optional[BaseImputer] = None,
		schema_policy: str = "error",
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
		random_state: Optional[int] = None,
		sampling_strategy: Optional[SamplingStrategy] = NoSampling(),
		x_scaler: Optional[Any] = "standard",
		y_scaler: Optional[Any] = "standard",
		splitter: Any = None,
		variable_descriptions: Optional[Mapping[str, VariableDescription]] = None,
		default_value_type: ValueType = ValueType.CONTINUOUS,
) -> PreparedData:
	from .pipeline import DataPreparer, DataPreparationConfig

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
		lag_selector_params=(
			{}
			if lag_selector_params is None
			else dict(lag_selector_params)
		),
		lag_n_jobs=lag_n_jobs,
		stationarity=stationarity,
		missing_value_imputer=missing_value_imputer,
		sampling_strategy=sampling_strategy,
		splitter=splitter,
		variable_descriptions=(
			{} if variable_descriptions is None else dict(variable_descriptions)
		),
		default_value_type=default_value_type,
	)
	preparer = DataPreparer(config=config)
	roles = None
	if causes is not None or tested_causes is not None or effects is not None:
		from .configuration.plan import SourceSchema

		schema = SourceSchema.from_frames(data, policy=config.schema_policy)
		cause_names = tuple(causes) if causes is not None else schema.columns
		effect_names = tuple(effects) if effects is not None else schema.columns
		tested_names = (
			tuple(tested_causes) if tested_causes is not None else cause_names
		)
		roles = VariableRoles(
			causes=cause_names,
			tested_causes=tested_names,
			effects=effect_names,
		)
	return preparer.fit(data, roles=roles).prepared_data_
