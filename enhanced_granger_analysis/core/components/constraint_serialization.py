from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .constraint_spec import (
    PreparedConstraintData,
    PreparedMinAbsSumRule,
)
from ..exceptions import ConstraintConfigurationError


def copy_constraint_data(
    data: PreparedConstraintData,
) -> PreparedConstraintData:
    if not isinstance(data, PreparedConstraintData):
        raise TypeError(
            "data must be a PreparedConstraintData instance."
        )

    return PreparedConstraintData(
        output_names=data.output_names,
        n_features=data.n_features,
        mask=data.mask,
        rules=data.rules,
    )


def constraint_data_to_config(
    data: PreparedConstraintData,
) -> dict[str, Any]:
    return {
        "output_names": list(data.output_names),
        "n_features": data.n_features,
        "mask": data.mask.tolist(),
        "rules": [
            {
                "output_index": rule.output_index,
                "feature_indices": list(rule.feature_indices),
                "min_abs_sum": rule.min_abs_sum,
            }
            for rule in data.rules
        ],
    }


def constraint_data_from_config(
    raw: Mapping[str, Any],
) -> PreparedConstraintData:
    if not isinstance(raw, Mapping):
        raise ConstraintConfigurationError(
            "Serialized constraint data must be a mapping."
        )

    required = {
        "output_names",
        "n_features",
        "mask",
        "rules",
    }

    if set(raw) != required:
        raise ConstraintConfigurationError(
            "Serialized constraint data must contain exactly: "
            "output_names, n_features, mask, rules."
        )

    output_names = raw["output_names"]
    if not isinstance(output_names, (list, tuple)):
        raise ConstraintConfigurationError(
            "Serialized output_names must be a list or tuple."
        )

    raw_rules = raw["rules"]
    if not isinstance(raw_rules, list):
        raise ConstraintConfigurationError(
            "Serialized rules must be a list."
        )

    rules = []
    rule_fields = {
        "output_index",
        "feature_indices",
        "min_abs_sum",
    }

    for index, raw_rule in enumerate(raw_rules):
        if (
            not isinstance(raw_rule, Mapping)
            or set(raw_rule) != rule_fields
        ):
            raise ConstraintConfigurationError(
                f"Serialized rules[{index}] must contain exactly: "
                "output_index, feature_indices, min_abs_sum."
            )

        feature_indices = raw_rule["feature_indices"]
        if not isinstance(feature_indices, (list, tuple)):
            raise ConstraintConfigurationError(
                f"Serialized rules[{index}].feature_indices "
                "must be a list or tuple."
            )

        rules.append(PreparedMinAbsSumRule(
            output_index=raw_rule["output_index"],
            feature_indices=tuple(feature_indices),
            min_abs_sum=raw_rule["min_abs_sum"],
        ))

    return PreparedConstraintData(
        output_names=tuple(output_names),
        n_features=raw["n_features"],
        mask=np.asarray(raw["mask"]),
        rules=tuple(rules),
    )