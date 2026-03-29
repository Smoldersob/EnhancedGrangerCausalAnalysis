from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import (
	ColumnMismatchError,
	DataShapeError,
	EmptyDataError,
	LagConfigurationError,
)


def validate_dataframe_list(
	data_list: Sequence[pd.DataFrame],
	*,
	require_same_columns: bool = True,
	require_same_shape: bool = False,
	allow_superset_columns: bool = False,
	copy: bool = False,
) -> Tuple[List[pd.DataFrame], List[str]]:
	"""Validate a list of DataFrames used in analysis stages.

	Parameters
	----------
	data_list:
		Sequence of pandas DataFrames.
	require_same_columns:
		If True, verify column compatibility across all DataFrames.
	require_same_shape:
		If True, enforce identical shape `(n_rows, n_cols)` for every DataFrame.
	allow_superset_columns:
		If True, each DataFrame may contain extra columns, but must include
		all columns from the first DataFrame. Returned DataFrames are reindexed
		to the first DataFrame's columns.
	copy:
		If True, return copies of validated/reindexed DataFrames.

	Returns
	-------
	validated, reference_columns
		Validated DataFrames and ordered reference column list.
	"""
	if not data_list:
		raise EmptyDataError("data_list must contain at least one DataFrame")

	first = data_list[0]
	if not isinstance(first, pd.DataFrame):
		raise ColumnMismatchError("Element at index 0 is not a pandas DataFrame")

	ref_columns = list(first.columns)
	ref_shape = tuple(first.shape)
	validated: List[pd.DataFrame] = []

	for idx, df in enumerate(data_list):
		if not isinstance(df, pd.DataFrame):
			raise ColumnMismatchError(
				f"Element at index {idx} is not a pandas DataFrame"
			)

		current = df

		if require_same_columns:
			if allow_superset_columns:
				missing = [col for col in ref_columns if col not in current.columns]
				if missing:
					raise ColumnMismatchError(
						f"DataFrame at index {idx} does not contain required columns: {missing}"
					)
				current = current[ref_columns]
			else:
				if list(current.columns) != ref_columns:
					raise ColumnMismatchError(
						f"DataFrame at index {idx} has columns {list(current.columns)}, "
						f"expected {ref_columns}"
					)

		if require_same_shape and tuple(current.shape) != ref_shape:
			raise DataShapeError(
				f"DataFrame at index {idx} has shape {tuple(current.shape)}, "
				f"expected {ref_shape}"
			)

		validated.append(current.copy() if copy else current)

	return validated, ref_columns


def validate_columns_present(
	available_columns: Iterable[str],
	required_columns: Iterable[str],
	*,
	context: str = "columns",
) -> None:
	"""Ensure all required columns are present in available columns."""
	available = set(available_columns)
	required = list(required_columns)
	missing = [col for col in required if col not in available]
	if missing:
		raise ColumnMismatchError(
			f"Missing required {context}: {missing}. Available: {sorted(available)}"
		)


def validate_lag_bounds(min_lags: np.ndarray, max_lags: np.ndarray) -> None:
	"""Validate lag bounds arrays for lag preparation stage."""
	min_lags = np.asarray(min_lags, dtype=int)
	max_lags = np.asarray(max_lags, dtype=int)

	if min_lags.shape != max_lags.shape:
		raise LagConfigurationError("min_lags and max_lags must have the same shape")
	if np.any(min_lags < 0) or np.any(max_lags < 0):
		raise LagConfigurationError("Lag values must be non-negative")
	if np.any(min_lags > max_lags):
		raise LagConfigurationError("Each min_lag must be <= corresponding max_lag")


__all__ = [
	"validate_dataframe_list",
	"validate_columns_present",
	"validate_lag_bounds",
]
