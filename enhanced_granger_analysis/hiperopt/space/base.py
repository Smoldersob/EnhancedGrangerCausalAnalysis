from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class Choices:
    values: tuple[Any, ...]


@dataclass(frozen=True)
class IntegerRange:
    low: int
    high: int


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    log: bool = False


Domain: TypeAlias = Choices | IntegerRange | FloatRange
ParameterPath: TypeAlias = tuple[str, ...]


@dataclass(frozen=True)
class SearchParameter:
    path: ParameterPath
    domain: Domain


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[SearchParameter, ...]


@dataclass(frozen=True)
class CandidateUpdate:
    path: ParameterPath
    value: Any


@dataclass(frozen=True)
class SearchCandidate:
    updates: tuple[CandidateUpdate, ...]