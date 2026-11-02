"""Prepare candidate configurations without constructing models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ..configuration.modeling_spec import ModelingSpec
from ..core.components.regularizer_parsing import (
    parse_regularizer_spec,
)
from ..hiperopt.space.application import apply_candidate
from ..hiperopt.space.base import SearchCandidate


def apply_modeling_candidate(
    modeling: ModelingSpec,
    candidate: SearchCandidate,
) -> ModelingSpec:
    """Return modeling settings with candidate updates applied."""
    regularizer = parse_regularizer_spec(
        modeling.regularizer_spec
    )

    configuration = apply_candidate(
        {
            "model_config": dict(modeling.model_config),
            "regularizer_spec": (
                None
                if regularizer is None
                else {
                    "type": regularizer.selector,
                    "params": dict(regularizer.params),
                }
            ),
        },
        candidate,
    )

    return replace(
        modeling,
        model_config=configuration["model_config"],
        regularizer_spec=configuration["regularizer_spec"],
    )


def apply_trial_epochs(
    modeling: ModelingSpec,
    epochs: int | None,
) -> ModelingSpec:
    """Apply a trial-only epoch limit without changing final settings."""
    if epochs is None:
        return modeling

    configuration = apply_candidate(
        {"model_config": dict(modeling.model_config)},
        SearchCandidate(updates=()),
    )["model_config"]

    # An explicit HPO budget replaces the baseline iteration limit.
    configuration.pop("max_iter", None)
    configuration["epochs"] = int(epochs)

    # Avoid duplicate constructor arguments from embedded model params.
    for selector in ("model", "model_type"):
        declaration = configuration.get(selector)

        if not isinstance(declaration, Mapping):
            continue

        params = declaration.get("params")
        if isinstance(params, Mapping):
            params = dict(params)
            params.pop("epochs", None)
            params.pop("max_iter", None)

            declaration = dict(declaration)
            declaration["params"] = params
            configuration[selector] = declaration

    return replace(modeling, model_config=configuration)