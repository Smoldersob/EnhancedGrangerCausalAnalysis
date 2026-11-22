from dataclasses import dataclass
from typing import Any, Literal


ParameterPath = tuple[str, ...]


class SearchDomain:
    """Base contract for normalized parameter domains."""


@dataclass(frozen=True)
class ChoiceDomain(SearchDomain):
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if not values:
            raise ValueError("A choice domain must contain a value.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SearchParameter:
    path: ParameterPath
    domain: SearchDomain

    def __post_init__(self) -> None:
        path = tuple(self.path)
        if (
            len(path) < 2
            or not all(
                isinstance(item, str) and item and item == item.strip()
                for item in path
            )
        ):
            raise ValueError("A parameter requires a valid section and path.")
        if not isinstance(self.domain, SearchDomain):
            raise TypeError("domain must implement SearchDomain.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[SearchParameter, ...]

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        paths = [parameter.path for parameter in parameters]
        for index, path in enumerate(paths):
            for other in paths[:index]:
                prefix = min(len(path), len(other))
                if path[:prefix] == other[:prefix]:
                    raise ValueError(
                        f"Conflicting parameter paths: {other!r}, {path!r}."
                    )
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True)
class CandidateUpdate:
    path: ParameterPath
    value: Any

    def __post_init__(self) -> None:
        path = tuple(self.path)
        if (
            len(path) < 2
            or not all(
                isinstance(item, str) and item and item == item.strip()
                for item in path
            )
        ):
            raise ValueError("Invalid candidate update path.")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class SearchCandidate:
    updates: tuple[CandidateUpdate, ...]

    def __post_init__(self) -> None:
        updates = tuple(self.updates)
        SearchSpace(tuple(
            SearchParameter(update.path, ChoiceDomain((update.value,)))
            for update in updates
        ))
        object.__setattr__(self, "updates", updates)
