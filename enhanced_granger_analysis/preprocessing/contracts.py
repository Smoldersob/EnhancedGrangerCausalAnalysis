"""Narrow integration contracts for optional preprocessing decision components.

These are interfaces only. They deliberately contain no diagnostic, policy,
or lag-search algorithms; callers must fail clearly until a concrete provider
is registered and connected to the fitting workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ..core.data.analysis_roles import VariableRoles
from .splitting.scope import FitScope


@dataclass(frozen=True)
class DiagnosticFact:
    fact_id: str
    kind: str
    variables: tuple[str, ...]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.fact_id or not self.kind or not self.variables:
            raise ValueError("Diagnostic facts require an id, kind, and variables.")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "evidence", dict(self.evidence))


@dataclass(frozen=True)
class DiagnosticsReport:
    scope: FitScope
    facts: tuple[DiagnosticFact, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(fact, DiagnosticFact) for fact in self.facts):
            raise TypeError("DiagnosticsReport.facts must contain DiagnosticFact values.")
        object.__setattr__(self, "facts", tuple(self.facts))


@dataclass(frozen=True)
class RepresentationAction:
    variable: str
    action: str
    fact_ids: tuple[str, ...] = ()
    source: str = "policy"
    representative: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"keep", "remove_predictor", "remove_variable"}:
            raise ValueError(f"Unsupported representation action {self.action!r}.")
        object.__setattr__(self, "fact_ids", tuple(self.fact_ids))


@dataclass(frozen=True)
class RepresentationDecision:
    actions: tuple[RepresentationAction, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))


class DiagnosticProvider(Protocol):
    def diagnose(self, frames: tuple[Any, ...], *, scope: FitScope) -> DiagnosticsReport: ...


class RepresentationPolicyResolver(Protocol):
    def resolve(
        self,
        roles: VariableRoles,
        report: DiagnosticsReport,
    ) -> RepresentationDecision: ...
__all__ = [
    "DiagnosticFact",
    "DiagnosticsReport",
    "RepresentationAction",
    "RepresentationDecision",
    "DiagnosticProvider",
    "RepresentationPolicyResolver",
]
