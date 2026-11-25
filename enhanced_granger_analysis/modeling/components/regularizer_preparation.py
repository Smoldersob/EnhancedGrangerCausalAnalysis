from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ...core.components.regularizers.layout import RegularizationLayout
from ...core.components.regularizers.prepared import PreparedRegularizerData
from ...core.components.regularizers.specification import RegularizerSpec


def prepare_regularization_layout(
    *,
    cause_slices: Mapping[str, slice],
    output_names: Sequence[str],
    feature_lags: Any,
    base_mask: Any = None,
    components: tuple[str | None, ...] = (),
) -> RegularizationLayout:
    output_names = tuple(output_names)
    feature_lags = np.asarray(feature_lags)

    if feature_lags.ndim != 1:
        raise ValueError("feature_lags must be one-dimensional.")

    if base_mask is None:
        base_mask = np.ones(
            (len(output_names), len(feature_lags)),
            dtype=bool,
        )

    return RegularizationLayout(
        cause_slices=cause_slices,
        output_names=output_names,
        feature_lags=feature_lags,
        base_mask=base_mask,
        components=components,
    )


def prepare_regularizer_data(
    *,
    layout: RegularizationLayout,
    spec: RegularizerSpec | None,
) -> PreparedRegularizerData | None:
    """
    Prepare constructor input without resolving or creating a class.
    """
    if spec is None:
        return None

    if not isinstance(spec, RegularizerSpec):
        raise TypeError(
            "spec must be normalized with parse_regularizer_spec()."
        )

    return PreparedRegularizerData(
        layout=layout,
        params=spec.params,
    )
