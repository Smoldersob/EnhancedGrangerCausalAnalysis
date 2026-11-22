from numbers import Integral, Real

import numpy as np

from ...exceptions import ConstraintConfigurationError
from .prepared import PreparedConstraintData, PreparedMinAbsSumRule


def validate_prepared_constraint_data(data: PreparedConstraintData) -> None:
    def integer(value):
        return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))

    def fail(message):
        raise ConstraintConfigurationError(message)

    if not integer(data.n_features) or data.n_features <= 0:
        fail("n_features must be a positive integer.")
    names = data.output_names
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        fail("output_names must contain non-empty strings.")
    if len(set(names)) != len(names):
        fail("output_names must be unique.")
    if data.mask.shape != (len(names), data.n_features):
        fail("Invalid canonical constraint mask shape.")

    seen = set()
    for index, rule in enumerate(data.rules):
        if not isinstance(rule, PreparedMinAbsSumRule):
            fail(f"Rule {index} must be a PreparedMinAbsSumRule.")
        output = rule.output_index
        features = rule.feature_indices
        minimum = rule.min_abs_sum
        if not integer(output) or not 0 <= output < len(names):
            fail(f"Rule {index}: invalid output_index.")
        if not features or any(
            not integer(feature) or not 0 <= feature < data.n_features
            for feature in features
        ):
            fail(f"Rule {index}: invalid feature_indices.")
        if len(set(features)) != len(features):
            fail(f"Rule {index}: repeated feature_indices.")
        if isinstance(minimum, (bool, np.bool_)) or not isinstance(minimum, Real):
            fail(f"Rule {index}: min_abs_sum must be real, not boolean.")
        if not np.isfinite(minimum) or minimum < 0:
            fail(f"Rule {index}: min_abs_sum must be finite and non-negative.")
        for feature in features:
            pair = (int(output), int(feature))
            if not data.mask[output, feature]:
                fail(f"Rule {index}: coefficient {pair} is hard-masked.")
            if pair in seen:
                fail(f"Rule {index}: overlapping min_abs_sum groups are not supported.")
            seen.add(pair)
