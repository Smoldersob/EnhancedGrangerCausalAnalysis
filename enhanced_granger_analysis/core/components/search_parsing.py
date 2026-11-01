"""Parse search declarations without constructing backend components."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..registry import Registry
from ..search import (
    ChoiceDomain,
    SearchComponentSpec,
    SearchDomain,
    SearchParameter,
    SearchSpace,
)

DomainParser = Callable[[Mapping[str, Any]], SearchDomain]

domain_parsers: Registry[DomainParser] = Registry("search domain")


def parse_search_component(spec: Any) -> SearchComponentSpec:
    """Normalize a name, class, instance, or explicit declaration."""

    if isinstance(spec, SearchComponentSpec):
        return spec

    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                f"Unknown component keys: {sorted(unknown, key=repr)!r}."
            )
        if "type" not in spec:
            raise ValueError("Component declaration requires 'type'.")
        selector = spec["type"]
        params = spec.get("params", {})
    else:
        selector, params = spec, {}

    if selector is None:
        raise ValueError("A search component must be selected.")

    return SearchComponentSpec(selector, params)


def parse_choice(params: Mapping[str, Any]) -> ChoiceDomain:
    """Parse explicit alternatives without interpreting their contents."""

    if set(params) != {"values"}:
        raise ValueError("Choice params must contain only 'values'.")

    values = params["values"]
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
    ):
        raise TypeError("Choice values must be a sequence.")

    return ChoiceDomain(tuple(values))


domain_parsers.register(
    "choice",
    parse_choice,
    aliases=("categorical",),
)


class SearchSpaceParser:
    """Parse explicit parameter entries using registered domain parsers."""

    def __init__(
        self,
        *,
        registry: Registry[DomainParser] | None = None,
    ) -> None:
        self.registry = domain_parsers if registry is None else registry

    def parse(self, declarations: Sequence[Mapping[str, Any]]) -> SearchSpace:
        """Read paths and domains without resolving destination components."""

        if (
            not isinstance(declarations, Sequence)
            or isinstance(declarations, (str, bytes))
        ):
            raise TypeError("Search space must be a sequence of entries.")

        parameters = []

        for entry in declarations:
            if not isinstance(entry, Mapping):
                raise TypeError("Each search entry must be a mapping.")
            if set(entry) != {"path", "domain"}:
                raise ValueError(
                    "Each search entry requires only 'path' and 'domain'."
                )

            path = entry["path"]
            if (
                not isinstance(path, Sequence)
                or isinstance(path, (str, bytes))
            ):
                raise TypeError("Parameter path must be a sequence of names.")

            declaration = parse_search_component(entry["domain"])
            selector = declaration.selector

            if not isinstance(selector, str):
                raise TypeError("Domain parser selection must be a name.")

            parser = self.registry.resolve(selector)
            domain = parser(declaration.params)

            parameters.append(SearchParameter(tuple(path), domain))

        return SearchSpace(tuple(parameters))