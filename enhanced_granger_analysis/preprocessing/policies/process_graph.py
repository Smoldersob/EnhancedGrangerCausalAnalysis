"""Opt-in, role-class process graph policy.

The policy emits ``RelationPolicyDecision`` values only. It does not edit data,
resolve representation changes, choose lags, or infer concrete process loops.
Because ``ProcessRelation.kind`` has no agreed graph semantics, this version
uses role classes rather than interpreting process associations. A missing
association is therefore unknown; only role-based class rules produce hard
exclusions.

``CONTROL_SIGNAL`` and ``MANIPULATED_VARIABLE`` are the regulator class.
``SETPOINT``/``MANUAL_SETPOINT`` and an unverified ``CONTROL_ERROR`` may
predict only regulator-class outputs. Regulator-class signals may predict only
``CONTROLLED_VARIABLE`` outputs. Other process roles add no restriction.
Rules constrain cross-signal directions only; a variable's own history is
left untouched. Multiple restrictive roles on a predictor intersect their
allowed target classes. A target with no declared process role is unknown and
is not excluded; a declared role outside the allowed classes is known and is
excluded.

A supported contemporaneous identity ``error = a - b`` causes the error
predictor to be excluded while retaining the defining relation in the
decision. The shared decision contract cannot remove a variable from all
representations, so this is deliberately not reported as full variable
removal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from ...core.data.analysis_roles import VariableRoles
from ...core.data.relations import (
    AlgebraicRelation,
    AlgebraicRelationEvidence,
    ProcessRelation,
    RelationPolicyDecision,
)
from ...core.data.variables import ProcessRole, VariableDescription


_REGULATOR_ROLES = frozenset(
    {ProcessRole.CONTROL_SIGNAL, ProcessRole.MANIPULATED_VARIABLE}
)
_SETPOINT_ROLES = frozenset({ProcessRole.SETPOINT, ProcessRole.MANUAL_SETPOINT})
_CONTROLLED_ROLES = frozenset({ProcessRole.CONTROLLED_VARIABLE})

# None means that a role does not introduce a process-graph restriction.
# Keeping this map explicit prevents roles such as FAULT or RESIDUAL from
# silently acquiring semantics through enum membership or naming conventions.
ROLE_TARGET_CLASSES: Mapping[ProcessRole, frozenset[ProcessRole] | None] = MappingProxyType(
    {
        ProcessRole.PROCESS_VARIABLE: None,
        ProcessRole.TERMINAL_OUTPUT: None,
        ProcessRole.REDUNDANT_MEASUREMENT: None,
        ProcessRole.VIRTUAL_MEASUREMENT: None,
        ProcessRole.SETPOINT: _REGULATOR_ROLES,
        ProcessRole.MANUAL_SETPOINT: _REGULATOR_ROLES,
        ProcessRole.CONTROL_SIGNAL: _CONTROLLED_ROLES,
        ProcessRole.MANIPULATED_VARIABLE: _CONTROLLED_ROLES,
        ProcessRole.CONTROL_ERROR: _REGULATOR_ROLES,
        ProcessRole.CONTROLLED_VARIABLE: None,
        ProcessRole.FAULT: None,
        ProcessRole.DIAGNOSTIC_SIGNAL: None,
        ProcessRole.RESIDUAL: None,
        ProcessRole.DISTURBANCE: None,
    }
)

_DecisionScope: TypeAlias = tuple[str, str]  # (effect/output, predictor)


class PolicyDecisionConflict(ValueError):
    """Raised when policy decisions prescribe different actions for one pair."""


def _is_supported_difference(
    evidence: AlgebraicRelationEvidence,
    error_name: str,
) -> bool:
    relation = evidence.relation
    return (
        evidence.status == "supported"
        and relation.target == error_name
        and relation.offset == 0
        and len(relation.terms) == 2
        and all(term.lag == 0 for term in relation.terms)
        and {term.coefficient for term in relation.terms} == {-1.0, 1.0}
    )


def _decision_scope(decision: RelationPolicyDecision) -> _DecisionScope | None:
    relation = decision.relation
    if isinstance(relation, ProcessRelation) and len(relation.signals) == 2:
        source, target = relation.signals
        predictor = decision.predictor or source
        if predictor == source:
            return target, predictor
    if isinstance(relation, AlgebraicRelation) and decision.predictor is not None:
        return relation.target, decision.predictor
    return None


def compose_policy_decisions(
    *decision_groups: Sequence[RelationPolicyDecision],
) -> tuple[RelationPolicyDecision, ...]:
    """Combine explicit decisions, rejecting contradictory actions per pair.

    Decisions are preserved in input order (including compatible duplicates)
    so their policy provenance is not lost. The function does not attempt to
    resolve lag-constraint intersections or choose between representation
    actions; such behavior is not present in the shared relation contract.
    """

    combined = tuple(decision for group in decision_groups for decision in group)
    actions: dict[_DecisionScope, str] = {}
    for decision in combined:
        scope = _decision_scope(decision)
        if scope is None:
            continue
        previous = actions.setdefault(scope, decision.action)
        if previous != decision.action:
            raise PolicyDecisionConflict(
                f"Conflicting policy actions for effect {scope[0]!r} and "
                f"predictor {scope[1]!r}: {previous!r} vs {decision.action!r}."
            )
    return combined


@dataclass(frozen=True)
class ProcessGraphPolicy:
    """Build opt-in class-based process restrictions from declarations/facts."""

    enabled: bool = False
    name: str = "process_graph"

    def resolve(
        self,
        descriptions: Mapping[str, VariableDescription],
        roles: VariableRoles,
        evidence: Sequence[AlgebraicRelationEvidence] = (),
    ) -> tuple[RelationPolicyDecision, ...]:
        """Return decisions for the declared output/predictor axes.

        ``evidence`` is intentionally a narrow explicit sequence, independent
        of any diagnostics report API. Only supported evidence for the exact
        difference form changes the CONTROL_ERROR fallback branch.
        """

        if not self.enabled:
            return ()

        if not isinstance(roles, VariableRoles):
            raise TypeError("roles must be a VariableRoles instance.")
        if not isinstance(descriptions, Mapping):
            raise TypeError("descriptions must map names to VariableDescription values.")
        for name, description in descriptions.items():
            if not isinstance(description, VariableDescription) or description.name != name:
                raise ValueError("Each description must be stored under its own variable name.")
        for fact in evidence:
            if not isinstance(fact, AlgebraicRelationEvidence):
                raise TypeError("evidence must contain AlgebraicRelationEvidence values.")

        confirmed_errors = {
            fact.relation.target
            for fact in evidence
            if fact.relation.target in descriptions
            and ProcessRole.CONTROL_ERROR
            in descriptions[fact.relation.target].process_roles
            and _is_supported_difference(fact, fact.relation.target)
        }

        decisions: list[RelationPolicyDecision] = []
        for error_name in sorted(confirmed_errors):
            for fact in evidence:
                if fact.relation.target != error_name or not _is_supported_difference(fact, error_name):
                    continue
                decisions.append(
                    RelationPolicyDecision(
                        relation=fact.relation,
                        policy=self.name,
                        action="exclude_predictor",
                        reason=(
                            "CONTROL_ERROR has supported contemporaneous difference "
                            "evidence; exclude the derived signal as a predictor and "
                            "retain its defining relation."
                        ),
                        strength="hard",
                        predictor=error_name,
                    )
                )

        outputs = tuple(roles.effects)
        predictors = tuple(roles.causes)
        for predictor in predictors:
            description = descriptions.get(predictor)
            if description is None:
                continue
            active_rules = [
                (role, allowed_targets)
                for role in description.process_roles
                if (allowed_targets := ROLE_TARGET_CLASSES[role]) is not None
                and not (role == ProcessRole.CONTROL_ERROR and predictor in confirmed_errors)
            ]
            if not active_rules:
                continue
            allowed_targets = set.intersection(
                *(set(targets) for _, targets in active_rules)
            )

            for output in outputs:
                if output == predictor:
                    continue
                target_description = descriptions.get(output)
                if target_description is None or not target_description.process_roles:
                    continue
                target_roles = frozenset(target_description.process_roles)
                if target_roles.isdisjoint(allowed_targets):
                    restricting_roles = ", ".join(
                        role.value for role, _ in active_rules
                    )
                    relation = ProcessRelation(
                        name=f"{self.name}:{predictor}->{output}",
                        kind="role_class_restriction",
                        signals=(predictor, output),
                    )
                    decisions.append(
                        RelationPolicyDecision(
                            relation=relation,
                            policy=self.name,
                            action="exclude_predictor",
                            reason=(
                                f"Outgoing influence from roles [{restricting_roles}] "
                                "is restricted to their declared target role classes."
                            ),
                            strength="hard",
                            predictor=predictor,
                        )
                    )

        return tuple(decisions)


__all__ = [
    "ProcessGraphPolicy",
    "PolicyDecisionConflict",
    "ROLE_TARGET_CLASSES",
    "compose_policy_decisions",
]
