from __future__ import annotations

from copy import deepcopy
from itertools import product
from math import exp, log
from random import Random

from ..space.base import (
    CandidateUpdate,
    Choices,
    FloatRange,
    IntegerRange,
    SearchCandidate,
)
from .base import BaseSearchStrategy
from .registry import register_search_strategy


def _candidate(parameters, values):
    return SearchCandidate(
        tuple(
            CandidateUpdate(parameter.path, deepcopy(value))
            for parameter, value in zip(parameters, values)
        )
    )


@register_search_strategy("grid", "grid_search")
class GridSearchStrategy(BaseSearchStrategy):
    """Enumerate all combinations of discrete parameter domains."""

    def candidates(self, space):
        domains = []

        for parameter in space.parameters:
            domain = parameter.domain

            if isinstance(domain, Choices):
                domains.append(domain.values)
            elif isinstance(domain, IntegerRange):
                domains.append(range(domain.low, domain.high + 1))
            else:
                raise ValueError(
                    "Grid search requires discrete domains. "
                    f"Use choices for {'.'.join(parameter.path)!r}."
                )

        for values in product(*domains):
            yield _candidate(space.parameters, values)


@register_search_strategy("random", "random_search")
class RandomSearchStrategy(BaseSearchStrategy):
    """Generate independent samples until the runner stops iteration."""

    def __init__(self, seed=None):
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer or None."
            )

        self.seed = seed

    def candidates(self, space):
        if not space.parameters:
            yield SearchCandidate(())
            return

        rng = Random(self.seed)

        while True:
            values = [
                self._sample(parameter.domain, rng)
                for parameter in space.parameters
            ]
            yield _candidate(space.parameters, values)

    @staticmethod
    def _sample(domain, rng):
        if isinstance(domain, Choices):
            return rng.choice(domain.values)

        if isinstance(domain, IntegerRange):
            return rng.randint(domain.low, domain.high)

        if isinstance(domain, FloatRange):
            if domain.low == domain.high:
                return domain.low

            if domain.log:
                return exp(
                    rng.uniform(log(domain.low), log(domain.high))
                )

            return rng.uniform(domain.low, domain.high)

        raise TypeError(
            f"Unsupported domain: {type(domain).__name__}."
        )