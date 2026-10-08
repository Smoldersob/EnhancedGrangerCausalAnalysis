import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ..core.exceptions import DataValidationError
from .lag import LagEngine, LagConfiguration, BaseLagSelector
from .stationarity import StationarityTransformer
from .scaling import IdentityScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, _BaseScaler


from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedData:
	stationarity_transformer: StationarityTransformer
	lag_engine: LagEngine
	col_offsets: np.ndarray
	x_scaler: Any
	y_scaler: Any
	X_scaled: np.ndarray
	y_scaled: np.ndarray
	X_raw: np.ndarray
	y_raw: np.ndarray
	base_mask: Optional[np.ndarray]
	causes: List[str]
	tested_causes: List[str]
	effects: List[str]
	removed_variable_info: dict[str, dict[str, Any]] = {}

def _to_dataframe_list(data: pd.DataFrame | Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
	if isinstance(data, pd.DataFrame):
		return [data.copy()]
	if isinstance(data, Sequence) and all(isinstance(df, pd.DataFrame) for df in data):
		return [df.copy() for df in data]
	raise DataValidationError("data must be a DataFrame or a sequence of DataFrames")

def _reduce_backend_load(
	X_scaled: np.ndarray,
	y_scaled: np.ndarray,
	X_raw: np.ndarray,
	y_raw: np.ndarray,
	backend_sample_fraction: float = 1.0,
	backend_max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Reduce training workload after scaling and before backend call."""
	if not 0.0 < backend_sample_fraction <= 1.0:
		raise ValueError("backend_sample_fraction must be in range (0.0, 1.0]")

	if backend_sample_fraction == 1.0 and backend_max_samples is None:
		return X_scaled, y_scaled, X_raw, y_raw

	n = X_scaled.shape[0]
	if n == 0:
		raise DataValidationError("No samples available after lag preparation")

	n_keep = max(1, int(np.floor(backend_sample_fraction * n)))
	if backend_max_samples is not None:
		if backend_max_samples <= 0:
			raise ValueError("backend_max_samples must be > 0")
		n_keep = min(n_keep, int(backend_max_samples))

	if n_keep >= n:
		return X_scaled, y_scaled, X_raw, y_raw

	idx = np.arange(n_keep, dtype=int)
	return X_scaled[idx], y_scaled[idx], X_raw[idx], y_raw[idx]


def _create_scaler(name: Optional[Any]) -> Any:
	if name is None:
		return IdentityScaler()

	if isinstance(name, str):
		name_lower = name.lower()
		if name_lower in {"identity", "none"}:
			return IdentityScaler()
		if name_lower in {"standard", "zscore"}:
			return StandardScaler()
		if name_lower in {"minmax"}:
			return MinMaxScaler()
		if name_lower in {"robust"}:
			return RobustScaler()
		if name_lower in {"maxabs"}:
			return MaxAbsScaler()
		raise ValueError(f"Unsupported scaler: {name}")

	if isinstance(name, type) and issubclass(name, _BaseScaler):
		return name()

	if isinstance(name, _BaseScaler):
		return name

	if all(hasattr(name, attr) and callable(getattr(name, attr)) for attr in ("fit_transform", "transform", "inverse_transform")):
		return name

	raise ValueError(f"Unsupported scaler: {name}")

def _align_and_remove_redundant_columns(
	data_list: Sequence[pd.DataFrame],
	causes: Optional[Sequence[str]] = None,
	tested_causes: Optional[Sequence[str]] = None,
	effects: Optional[Sequence[str]] = None,
	fill_value: float = 0.0,
) -> tuple[
	list[pd.DataFrame],
	list[str],
	list[str],
	list[str],
	list[str],
	dict[str, str],
	dict[str, dict[str, str | list[str]]],
]:
	"""
	Align DataFrames using the union of all columns, remove globally
	constant columns and globally duplicated columns, and update variable lists.

	Parameters
	----------
	data_list : Sequence[pd.DataFrame]
		Input DataFrames.

	causes : Optional[Sequence[str]]
		Variables considered as possible causes. If None, all columns are used.

	tested_causes : Optional[Sequence[str]]
		Variables explicitly selected for testing. If None, all causes are used.

	effects : Optional[Sequence[str]]
		Variables considered as effects. If None, all columns are used.

	fill_value : float, default=0.0
		Value inserted when a column is absent from one DataFrame.

	Returns
	-------
	cleaned_data_list : list[pd.DataFrame]
		Aligned DataFrames with constant and duplicate columns removed.

	causes_list : list[str]
		Updated causes list.

	tested_causes_list : list[str]
		Updated tested causes list.

	effects_list : list[str]
		Updated effects list.

	removed_constant_columns : list[str]
		Names of globally constant columns removed from the DataFrames.

	duplicate_columns_map : dict[str, str]
		Mapping:
		{removed_duplicate_column: retained_identical_column}.

	removed_variable_info : dict[str, dict[str, str | list[str]]]
		Metadata saved only for removed variables that appeared in effects
		and/or tested_causes. For duplicate variables, the retained equivalent
		column is also stored.

	Notes
	-----
	A column appearing only in `causes` is removed from causes_list but is not
	written to removed_variable_info.
	"""
	if not data_list:
		raise DataValidationError("The input data list is empty.")

	if not all(isinstance(df, pd.DataFrame) for df in data_list):
		raise DataValidationError(
			"Every element of data_list must be a pandas DataFrame."
		)

	# Reject duplicated column labels before reindexing.
	for table_index, dataframe in enumerate(data_list):
		if dataframe.columns.duplicated().any():
			duplicated_names = dataframe.columns[
				dataframe.columns.duplicated()
			].tolist()

			raise DataValidationError(
				f"DataFrame at index {table_index} contains duplicated "
				f"column names: {duplicated_names}"
			)

	# Build an ordered union of all columns.
	# Columns from earlier DataFrames retain priority and order.
	all_columns = list(
		dict.fromkeys(
			column
			for dataframe in data_list
			for column in dataframe.columns
		)
	)

	# Validate user-provided names against the union before removing columns.
	causes_list = list(causes) if causes is not None else all_columns.copy()
	effects_list = list(effects) if effects is not None else all_columns.copy()
	tested_causes_list = (
		list(tested_causes)
		if tested_causes is not None
		else causes_list.copy()
	)

	unknown_causes = [
		column for column in causes_list if column not in all_columns
	]
	unknown_effects = [
		column for column in effects_list if column not in all_columns
	]
	unknown_tested_causes = [
		column for column in tested_causes_list if column not in all_columns
	]

	if unknown_causes or unknown_effects or unknown_tested_causes:
		raise DataValidationError(
			"Unknown variables detected. "
			f"causes={unknown_causes}, "
			f"effects={unknown_effects}, "
			f"tested_causes={unknown_tested_causes}"
		)

	# Align every DataFrame to the column union.
	# Newly created columns are filled with fill_value.
	aligned_data_list = []

	for table_index, dataframe in enumerate(data_list):
		try:
			aligned_dataframe = dataframe.reindex(
				columns=all_columns,
				fill_value=fill_value,
			).astype(float)

		except (TypeError, ValueError) as error:
			raise DataValidationError(
				f"Cannot convert DataFrame at index {table_index} "
				f"to float: {error}"
			) from error

		aligned_data_list.append(aligned_dataframe)

	# Concatenate only for global feature diagnostics.
	combined_dataframe = pd.concat(
		aligned_data_list,
		axis=0,
		ignore_index=True,
	)

	# NaN is deliberately treated as an actual value.
	# Thus [NaN, NaN] is considered constant, while [0.0, NaN] is not.
	removed_constant_columns = [
		column
		for column in all_columns
		if combined_dataframe[column].nunique(dropna=False) <= 1
	]

	data_without_constants = [
		dataframe.drop(columns=removed_constant_columns)
		for dataframe in aligned_data_list
	]

	combined_without_constants = pd.concat(
		data_without_constants,
		axis=0,
		ignore_index=True,
	)

	# Retain the first occurrence and map every later identical column
	# to the retained one.
	duplicate_columns_map: dict[str, str] = {}
	retained_columns: list[str] = []

	for column in combined_without_constants.columns:
		retained_match = None

		for retained_column in retained_columns:
			if combined_without_constants[column].equals(
				combined_without_constants[retained_column]
			):
				retained_match = retained_column
				break

		if retained_match is None:
			retained_columns.append(column)
		else:
			duplicate_columns_map[column] = retained_match

	duplicate_columns_to_remove = list(duplicate_columns_map)

	cleaned_data_list = [
		dataframe.drop(columns=duplicate_columns_to_remove)
		for dataframe in data_without_constants
	]

	# Keep metadata only for columns that were effects and/or tested causes.
	removed_variable_info: dict[str, dict[str, str | list[str]]] = {}

	for column in removed_constant_columns:
		roles: list[str] = []

		if column in effects_list:
			roles.append("effect")

		if column in tested_causes_list:
			roles.append("tested_cause")

		if roles:
			removed_variable_info[column] = {
				"reason": "constant",
				"roles": roles,
			}

	for duplicate_column, retained_column in duplicate_columns_map.items():
		roles = []

		if duplicate_column in effects_list:
			roles.append("effect")

		if duplicate_column in tested_causes_list:
			roles.append("tested_cause")

		if roles:
			removed_variable_info[duplicate_column] = {
				"reason": "duplicate",
				"retained_column": retained_column,
				"roles": roles,
			}

	# Remove all eliminated variables from every role list.
	removed_columns = set(removed_constant_columns)
	removed_columns.update(duplicate_columns_to_remove)

	causes_list = [
		column for column in causes_list
		if column not in removed_columns
	]

	tested_causes_list = [
		column for column in tested_causes_list
		if column not in removed_columns
	]

	effects_list = [
		column for column in effects_list
		if column not in removed_columns
	]

	return (
		cleaned_data_list,
		causes_list,
		tested_causes_list,
		effects_list,
		removed_variable_info,
	)
	  
def prepare_data(
		data: pd.DataFrame | Sequence[pd.DataFrame],
		causes: Optional[Sequence[str]] = None,
		tested_causes: Optional[Sequence[str]] = None,
		effects: Optional[Sequence[str]] = None,
		lag_config: Optional[LagConfiguration] = None,
		lag_selector: Optional[BaseLagSelector] = None,
		stationarity_transformer: Optional[StationarityTransformer] = None,
		backend_sample_fraction: float = 1.0,
		backend_max_samples: Optional[int] = None,
		x_scaler: Optional[Any] = "standard",
		y_scaler: Optional[Any] = "standard",
) -> PreparedData:
	# Column validation and consistency enforment
	data_list = _to_dataframe_list(data)

	(
		data_list,
		causes_list,
		tested_causes_list,
		effects_list,
		removed_variable_info,
	) = _align_and_remove_redundant_columns(
		data_list=data_list,
		causes=causes,
		tested_causes=tested_causes,
		effects=effects,
		fill_value=0.0,
	)

	# These are the columns that remain after alignment and feature removal.
	all_columns = data_list[0].columns.tolist()

	if len(all_columns) == 0:
		raise DataValidationError(
			"No columns remain after removing constant and duplicate columns."
		)

	if len(effects_list) == 0:
		raise DataValidationError(
			"No effect variables remain after removing constant and "
			"duplicate columns."
		)

	if len(tested_causes_list) == 0:
		raise DataValidationError(
			"No tested cause variables remain after removing constant and "
			"duplicate columns."
		)

	# Stationarity transformation
	stationarity = stationarity_transformer
	data_stationary = stationarity.fit_transform(data_list)

	# Lag preparation
	engine = LagEngine(config=lag_config or LagConfiguration(), selector=lag_selector)
	X_train, y_train, col_offsets = engine.prepare(data_stationary, effects=effects_list)

	# scaling
	x_scaler_obj = _create_scaler(x_scaler)
	y_scaler_obj = _create_scaler(y_scaler)

	X_train_scaled = x_scaler_obj.fit_transform(X_train)
	y_train_scaled = y_scaler_obj.fit_transform(y_train)

	# Reduce backend load (by droping samples) if requested
	X_backend_scaled, y_backend_scaled, X_backend_raw, y_backend_raw = _reduce_backend_load(
		X_scaled=X_train_scaled,
		y_scaled=y_train_scaled,
		X_raw=X_train,
		y_raw=y_train,
		backend_sample_fraction=backend_sample_fraction,
		backend_max_samples=backend_max_samples,
	)

	base_mask = None
	if engine.mask_ is not None:
		mask_arr = np.asarray(engine.mask_, dtype=np.float64)
		if mask_arr.shape[0] == len(effects_list):
			base_mask = mask_arr
		else:
			effect_indices = [all_columns.index(effect_name) for effect_name in effects_list]
			base_mask = mask_arr[effect_indices, :]

	prepared = PreparedData(
		stationarity_transformer=stationarity,
		lag_engine=engine,
		col_offsets=col_offsets,
		x_scaler=x_scaler_obj,
		y_scaler=y_scaler_obj,
		X_scaled=X_backend_scaled,
		y_scaled=y_backend_scaled,
		X_raw=X_backend_raw,
		y_raw=y_backend_raw,
		base_mask=base_mask,
		causes=causes_list,
		tested_causes=tested_causes_list,
		effects=effects_list,
        removed_variable_info=removed_variable_info,
	)

	return prepared