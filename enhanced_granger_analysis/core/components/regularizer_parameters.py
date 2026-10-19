from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..exceptions import RegularizerConfigurationError
from .regularizer_spec import PreparedRegularizerData


L1_DEFAULT_PARAMS = {
    "l1": 0.0,
}

LAG_L1_DEFAULT_PARAMS = {
    "l1": 0.0,
    "lag_weights": tuple(float(i) for i in range(1, 21)),
    "relative_lag_indexing": False,
}


def _resolve_params(
    data: PreparedRegularizerData,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(data, PreparedRegularizerData):
        raise TypeError("data must be PreparedRegularizerData.")

    unknown = set(data.params) - set(defaults)

    if unknown:
        raise RegularizerConfigurationError(
            f"Unknown regularizer parameters: {sorted(unknown)!r}."
        )

    return {**defaults, **data.params}


def _validate_l1(value: Any) -> float:
    raw = np.asarray(value)

    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise RegularizerConfigurationError(
            "l1 must be a real numeric scalar."
        )

    result = float(raw)

    if not np.isfinite(result) or result < 0:
        raise RegularizerConfigurationError(
            "l1 must be finite and >= 0."
        )

    return result


def prepare_l1_data(
    data: PreparedRegularizerData,
) -> PreparedRegularizerData:
    params = _resolve_params(data, L1_DEFAULT_PARAMS)

    return PreparedRegularizerData(
        layout=data.layout,
        params={
            "l1": _validate_l1(params["l1"]),
        },
    )


def prepare_lag_l1_data(
    data: PreparedRegularizerData,
) -> tuple[
    PreparedRegularizerData,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Return:
        effective_data,
        lag_weight_indices,
        feature_weights,
        lag_origin.
    """
    params = _resolve_params(data, LAG_L1_DEFAULT_PARAMS)
    l1 = _validate_l1(params["l1"])

    relative = params["relative_lag_indexing"]

    if not isinstance(relative, (bool, np.bool_)):
        raise RegularizerConfigurationError(
            "relative_lag_indexing must be boolean."
        )

    relative = bool(relative)
    raw_weights = np.asarray(params["lag_weights"])

    if (
        raw_weights.ndim != 1
        or raw_weights.size == 0
        or raw_weights.dtype.kind not in "iuf"
    ):
        raise RegularizerConfigurationError(
            "lag_weights must be a non-empty 1D sequence "
            "of real numbers."
        )

    lag_weights = np.array(
        raw_weights,
        dtype=np.float64,
        copy=True,
    )

    if (
        not np.all(np.isfinite(lag_weights))
        or np.any(lag_weights < 0)
    ):
        raise RegularizerConfigurationError(
            "lag_weights must be finite and non-negative."
        )

    feature_lags = data.layout.feature_lags
    lag_origin = int(feature_lags.min()) if relative else 0
    indices = feature_lags - lag_origin

    required_length = int(indices.max()) + 1

    if lag_weights.size < required_length:
        raise RegularizerConfigurationError(
            f"lag_weights requires at least {required_length} "
            f"entries for relative_lag_indexing={relative}; "
            f"got {lag_weights.size}."
        )

    effective_data = PreparedRegularizerData(
        layout=data.layout,
        params={
            "l1": l1,
            "lag_weights": lag_weights.tolist(),
            "relative_lag_indexing": relative,
        },
    )

    return (
        effective_data,
        indices.copy(),
        lag_weights[indices].copy(),
        lag_origin,
    )