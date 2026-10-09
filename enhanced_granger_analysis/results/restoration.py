from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .causality_matrix import CausalityMatrix


def restore_removed_matrix(
    matrix: CausalityMatrix | pd.DataFrame,
    *,
    effects: Iterable[str],
    causes: Iterable[str],
    removed_variables: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Restore removed variables using the value declared by the metric."""
    if not removed_variables:
        return matrix.data.copy() if isinstance(matrix, CausalityMatrix) else matrix.copy()

    frame = matrix.data if isinstance(matrix, CausalityMatrix) else matrix
    metadata = matrix.metadata if isinstance(matrix, CausalityMatrix) else {}
    missing_value = metadata.get("removed_signal_value", np.nan)
    full_effects = list(dict.fromkeys(effects))
    full_causes = list(dict.fromkeys(causes))
    full_effects.extend(
        name for name, info in removed_variables.items()
        if "effect" in info.get("roles", []) and name not in full_effects
    )
    full_causes.extend(
        name for name, info in removed_variables.items()
        if "tested_cause" in info.get("roles", []) and name not in full_causes
    )

    result = pd.DataFrame(missing_value, index=full_effects, columns=full_causes, dtype=np.float64)
    result.loc[frame.index.intersection(result.index), frame.columns.intersection(result.columns)] = frame.loc[
        frame.index.intersection(result.index), frame.columns.intersection(result.columns)
    ]

    for removed_name, info in removed_variables.items():
        roles = info.get("roles", [])
        retained = info.get("retained_column")
        if info.get("reason") != "duplicate":
            continue
        if "effect" in roles and retained in result.index and removed_name in result.index:
            result.loc[removed_name, :] = result.loc[retained, :].to_numpy(dtype=np.float64)
        if "tested_cause" in roles and retained in result.columns and removed_name in result.columns:
            result.loc[:, removed_name] = result.loc[:, retained].to_numpy(dtype=np.float64)
        if (
            "effect" in roles
            and "tested_cause" in roles
            and retained in result.index
            and retained in result.columns
            and removed_name in result.index
            and removed_name in result.columns
        ):
            result.loc[removed_name, removed_name] = result.loc[retained, retained]

    return result
