from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .causality_matrix import CausalityMatrix

if TYPE_CHECKING:
    from ..preprocessing.prepare_data import RemovedVariable


def restore_removed_matrix(
    matrix: CausalityMatrix | pd.DataFrame,
    *,
    effects: Iterable[str],
    causes: Iterable[str],
    removed_variables: Mapping[str, RemovedVariable],
) -> pd.DataFrame:
    """Restore requested axes using role-specific removal metadata.

    Lag-excluded tested causes receive the metric's neutral removal
    value. This denotes structural exclusion by the selected model,
    not an independently computed hypothesis-test result.
    """
    frame = (
        matrix.data
        if isinstance(matrix, CausalityMatrix)
        else matrix
    )
    metadata = (
        matrix.metadata
        if isinstance(matrix, CausalityMatrix)
        else {}
    )

    missing_value = metadata.get(
        "removed_signal_value",
        np.nan,
    )

    full_effects = list(effects)
    full_causes = list(causes)

    if len(full_effects) != len(set(full_effects)):
        raise ValueError(
            "Requested effects contain duplicated names."
        )

    if len(full_causes) != len(set(full_causes)):
        raise ValueError(
            "Requested tested causes contain duplicated names."
        )

    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(
            "The input matrix must have unique row and column names."
        )

    result = frame.reindex(
        index=full_effects,
        columns=full_causes,
        fill_value=missing_value,
    ).astype(np.float64)

    # Restore duplicate effects from computed representative rows.
    for removed_name, info in removed_variables.items():
        if (
            info.reason == "duplicate"
            and "effect" in info.removed_roles
            and removed_name in result.index
            and info.retained_variable in frame.index
            and info.retained_variable in result.index
        ):
            result.loc[removed_name, :] = result.loc[
                info.retained_variable, :
            ].to_numpy(copy=True)

    # Restore lag-excluded tested causes as neutral columns.
    # Do not touch rows: these variables may remain valid effects.
    lag_excluded_causes: set[str] = set()

    for removed_name, info in removed_variables.items():
        if (
            info.reason == "no_selected_lags"
            and "tested_cause" in info.removed_roles
            and removed_name in result.columns
        ):
            result.loc[:, removed_name] = missing_value
            lag_excluded_causes.add(removed_name)

    # Duplicate tested causes can copy either a computed column or
    # a representative restored after lag exclusion.
    available_cause_columns = (
        set(frame.columns) | lag_excluded_causes
    )

    for removed_name, info in removed_variables.items():
        representative = info.retained_variable

        if (
            info.reason == "duplicate"
            and "tested_cause" in info.removed_roles
            and removed_name in result.columns
            and representative in available_cause_columns
            and representative in result.columns
        ):
            result.loc[:, removed_name] = result.loc[
                :, representative
            ].to_numpy(copy=True)

    return result