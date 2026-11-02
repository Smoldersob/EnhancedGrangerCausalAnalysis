from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from numbers import Integral, Real
from typing import Any

from .base import (
    Choices,
    Domain,
    FloatRange,
    IntegerRange,
    SearchParameter,
    SearchSpace,
)


class SearchSpaceParser:
    """Parse parameter paths and domains without building components."""

    _ROOTS = {"model_config", "regularizer_spec"}

    _RESERVED_MODEL_FIELDS = {
        "input_spec",
        "n_features",
        "n_outputs",
        "components",
        "constraint",
        "constraint_spec",
        "regularizer",
        "regularizer_spec",
        "callbacks",
    }

    def parse(self, spec: Mapping[str, Any]) -> SearchSpace:
        """Parse a mapping of parameter paths to domain declarations."""
        if not isinstance(spec, Mapping):
            raise TypeError("Search space must be a mapping.")

        parameters = []
        paths = []

        for name, declaration in spec.items():
            path = self._parse_path(name)

            for previous in paths:
                common = min(len(path), len(previous))
                if path[:common] == previous[:common]:
                    raise ValueError(
                        f"Conflicting parameter paths: "
                        f"{'.'.join(previous)!r} and {name!r}."
                    )

            domain = self._parse_domain(declaration, name)
            parameters.append(SearchParameter(path, domain))
            paths.append(path)

        return SearchSpace(tuple(parameters))

    def _parse_path(self, name: Any) -> tuple[str, ...]:
        if not isinstance(name, str):
            raise TypeError("Parameter paths must be strings.")

        path = tuple(name.split("."))

        if len(path) < 2 or any(
            not part or part != part.strip()
            for part in path
        ):
            raise ValueError(f"Invalid parameter path: {name!r}.")

        if path[0] not in self._ROOTS:
            raise ValueError(
                f"Unsupported search-space root: {path[0]!r}."
            )

        if (
            path[0] == "model_config"
            and path[1] in self._RESERVED_MODEL_FIELDS
        ):
            raise ValueError(
                f"Field {path[1]!r} is managed outside model_config."
            )

        return path

    def _parse_domain(self, spec: Any, path: str) -> Domain:
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"Domain for {path!r} must be a mapping."
            )

        if "choices" in spec:
            self._check_fields(spec, {"choices"}, path)
            values = spec["choices"]

            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(
                    f"Domain {path!r} requires non-empty choices."
                )

            return Choices(
                tuple(deepcopy(value) for value in values)
            )

        kind = spec.get("type")

        if kind == "int":
            self._check_fields(
                spec, {"type", "low", "high"}, path
            )
            low = self._integer(spec.get("low"), path)
            high = self._integer(spec.get("high"), path)

            if low > high:
                raise ValueError(
                    f"Domain {path!r} requires low <= high."
                )

            return IntegerRange(low, high)

        if kind == "float":
            self._check_fields(
                spec, {"type", "low", "high", "log"}, path
            )
            low = self._real(spec.get("low"), path)
            high = self._real(spec.get("high"), path)
            log = spec.get("log", False)

            if not isinstance(log, bool):
                raise TypeError(
                    f"Domain {path!r}: log must be bool."
                )

            if low > high or (log and low <= 0):
                raise ValueError(
                    f"Invalid bounds for domain {path!r}."
                )

            return FloatRange(low, high, log)

        raise ValueError(
            f"Unknown domain type for {path!r}: {kind!r}."
        )

    @staticmethod
    def _check_fields(spec, allowed, path):
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(
                f"Unknown fields for {path!r}: "
                f"{sorted(unknown, key=repr)!r}."
            )

    @staticmethod
    def _integer(value, path):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"Domain {path!r} requires integer bounds."
            )
        return int(value)

    @staticmethod
    def _real(value, path):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Domain {path!r} requires numeric bounds."
            )

        result = float(value)
        if not isfinite(result):
            raise ValueError(
                f"Domain {path!r} requires finite bounds."
            )
        return result