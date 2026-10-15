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
    """Restore requested axes using typed removal metadata.

    effects:
        Requested output variables, in their original order.
    causes:
        Requested tested causes, in their original order.
    removed_variables:
        Mapping of removed variable names to their removal metadata.
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

    missing_value = metadata.get("removed_signal_value", np.nan)
    full_effects = list(effects)
    full_causes = list(causes)

    if len(full_effects) != len(set(full_effects)):
        raise ValueError("Requested effects contain duplicated names.")
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
        ):
            result.loc[removed_name, :] = result.loc[
                info.retained_variable, :
            ].to_numpy(copy=True)

    # Restore duplicate tested causes after restoring effect rows.
    # A representative that was only an input has no computed test column.
    for removed_name, info in removed_variables.items():
        if (
            info.reason == "duplicate"
            and "tested_cause" in info.removed_roles
            and removed_name in result.columns
            and info.retained_variable in frame.columns
        ):
            result.loc[:, removed_name] = result.loc[
                :, info.retained_variable
            ].to_numpy(copy=True)

    return result