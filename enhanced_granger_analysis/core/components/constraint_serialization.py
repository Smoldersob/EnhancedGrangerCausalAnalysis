from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .constraint_spec import (
    PreparedConstraintData,
    PreparedMinAbsSumRule,
)
from ..exceptions import ConstraintConfigurationError



def validate_prepared_constraint_data(data: PreparedConstraintData) -> None:
    # Disjoint groups are the contract already required by Keras.
    from numbers import Integral, Real
    def integer(value):
        return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
    def fail(message):
        raise ConstraintConfigurationError(message)
    if not integer(data.n_features) or data.n_features <= 0:
        fail('n_features must be a positive integer.')
    names = data.output_names
    if not names or any(not isinstance(n, str) or not n.strip() for n in names):
        fail('output_names must contain non-empty strings.')
    if len(set(names)) != len(names):
        fail('output_names must be unique.')
    if data.mask.shape != (len(names), data.n_features):
        fail('Invalid canonical constraint mask shape.')
    seen = set()
    for i, rule in enumerate(data.rules):
        if not isinstance(rule, PreparedMinAbsSumRule):
            fail(f'Rule {i} must be a PreparedMinAbsSumRule.')
        output, features, minimum = rule.output_index, rule.feature_indices, rule.min_abs_sum
        if not integer(output) or not 0 <= output < len(names):
            fail(f'Rule {i}: invalid output_index.')
        if not features or any(not integer(f) or not 0 <= f < data.n_features for f in features):
            fail(f'Rule {i}: invalid feature_indices.')
        if len(set(features)) != len(features):
            fail(f'Rule {i}: repeated feature_indices.')
        if isinstance(minimum, (bool, np.bool_)) or not isinstance(minimum, Real):
            fail(f'Rule {i}: min_abs_sum must be real, not boolean.')
        if not np.isfinite(minimum) or minimum < 0:
            fail(f'Rule {i}: min_abs_sum must be finite and non-negative.')
        for feature in features:
            pair = (int(output), int(feature))
            if not data.mask[output, feature]:
                fail(f'Rule {i}: coefficient {pair} is hard-masked.')
            if pair in seen:
                fail(f'Rule {i}: overlapping min_abs_sum groups are not supported.')
            seen.add(pair)


def copy_constraint_data(
    data: PreparedConstraintData,
) -> PreparedConstraintData:
    if not isinstance(data, PreparedConstraintData):
        raise TypeError(
            "data must be a PreparedConstraintData instance."
        )

    validate_prepared_constraint_data(data)

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