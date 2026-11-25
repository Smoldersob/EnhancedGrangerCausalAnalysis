import pytest

from ..core.data.analysis_roles import VariableRoles
from ..core.data.relations import (
    AlgebraicRelation,
    AlgebraicRelationEvidence,
    AlgebraicTerm,
    ProcessRelation,
    RelationPolicyDecision,
)
from ..core.data.variables import ProcessRole, ValueType, VariableDescription
from ..preprocessing.policies import (
    PolicyDecisionConflict,
    ProcessGraphPolicy,
    compose_policy_decisions,
)


def _description(name, *roles):
    return VariableDescription(name, ValueType.CONTINUOUS, tuple(roles))


def _pair_decision(source, target, action):
    relation = ProcessRelation("explicit-pair", "role_class_restriction", (source, target))
    return RelationPolicyDecision(
        relation,
        "test",
        action,
        "test decision",
        predictor=source if action == "exclude_predictor" else None,
    )


def _resolve(descriptions, effects, causes, evidence=(), *, enabled=True):
    return ProcessGraphPolicy(enabled=enabled).resolve(
        descriptions,
        VariableRoles(tuple(effects), tuple(causes), ()),
        evidence,
    )


def test_disabled_policy_has_no_decisions_even_with_role_and_evidence():
    relation = AlgebraicRelation(
        "error",
        (AlgebraicTerm("setpoint", 1), AlgebraicTerm("measurement", -1)),
    )
    evidence = AlgebraicRelationEvidence(relation, "test", "verification", "supported")

    assert _resolve(
        {
            "error": _description("error", ProcessRole.CONTROL_ERROR),
            "setpoint": _description("setpoint", ProcessRole.SETPOINT),
            "measurement": _description("measurement", ProcessRole.PROCESS_VARIABLE),
        },
        ("measurement",),
        ("error",),
        (evidence,),
        enabled=False,
    ) == ()


def test_supported_difference_excludes_error_predictor_and_keeps_relation():
    relation = AlgebraicRelation(
        "error",
        (AlgebraicTerm("setpoint", 1), AlgebraicTerm("measurement", -1)),
    )
    evidence = AlgebraicRelationEvidence(relation, "verifier", "verification", "supported")

    decisions = _resolve(
        {
            "error": _description("error", ProcessRole.CONTROL_ERROR),
            "actuator": _description("actuator", ProcessRole.MANIPULATED_VARIABLE),
            "process": _description("process", ProcessRole.PROCESS_VARIABLE),
        },
        ("actuator", "process"),
        ("error",),
        (evidence,),
    )

    assert len(decisions) == 1
    assert decisions[0].relation == relation
    assert decisions[0].action == "exclude_predictor"
    assert decisions[0].predictor == "error"
    assert "retain its defining relation" in decisions[0].reason


@pytest.mark.parametrize(
    "status,method,offset,coefficient,lag",
    [
        ("inconclusive", "verification", 0, -1, 0),
        ("contradicted", "verification", 0, -1, 0),
        ("supported", "detection", 1, -1, 0),
        ("supported", "verification", 0, -2, 0),
        ("supported", "verification", 0, -1, 1),
    ],
)
def test_unconfirmed_or_non_difference_evidence_uses_fallback(
    status, method, offset, coefficient, lag
):
    relation = AlgebraicRelation(
        "error",
        (
            AlgebraicTerm("setpoint", 1),
            AlgebraicTerm("measurement", coefficient, lag),
        ),
        offset=offset,
    )
    evidence = AlgebraicRelationEvidence(relation, "checker", method, status)
    descriptions = {
        "error": _description("error", ProcessRole.CONTROL_ERROR),
        "actuator": _description("actuator", ProcessRole.CONTROL_SIGNAL),
        "plant": _description("plant", ProcessRole.PROCESS_VARIABLE),
    }

    decisions = _resolve(descriptions, ("actuator", "plant"), ("error",), (evidence,))
    pairs = {(decision.relation.signals[0], decision.relation.signals[1]) for decision in decisions}

    assert pairs == {("error", "plant")}


def test_missing_difference_evidence_is_fallback_not_proof_of_no_relation():
    descriptions = {
        "error": _description("error", ProcessRole.CONTROL_ERROR),
        "actuator": _description("actuator", ProcessRole.MANIPULATED_VARIABLE),
        "plant": _description("plant", ProcessRole.CONTROLLED_VARIABLE),
    }

    decisions = _resolve(descriptions, ("actuator", "plant"), ("error",))

    assert [(decision.relation.signals, decision.action) for decision in decisions] == [
        (("error", "plant"), "exclude_predictor")
    ]


def test_setpoint_and_regulator_rules_and_own_history():
    descriptions = {
        "setpoint": _description("setpoint", ProcessRole.MANUAL_SETPOINT),
        "controller": _description("controller", ProcessRole.CONTROL_SIGNAL),
        "valve": _description("valve", ProcessRole.MANIPULATED_VARIABLE),
        "plant": _description("plant", ProcessRole.CONTROLLED_VARIABLE),
        "other": _description("other", ProcessRole.PROCESS_VARIABLE),
    }
    decisions = _resolve(
        descriptions,
        ("setpoint", "controller", "valve", "plant", "other"),
        ("setpoint", "controller", "valve"),
    )
    pairs = {
        decision.relation.signals
        for decision in decisions
        if isinstance(decision.relation, ProcessRelation)
    }

    assert ("setpoint", "plant") in pairs
    assert ("setpoint", "other") in pairs
    assert ("controller", "setpoint") in pairs
    assert ("controller", "valve") in pairs
    assert ("controller", "other") in pairs
    assert ("valve", "other") in pairs
    assert not any(source == target for source, target in pairs)
    assert ("controller", "plant") not in pairs
    assert ("valve", "plant") not in pairs


def test_multiple_roles_intersect_target_classes_and_keep_unclassified_roles_neutral():
    descriptions = {
        "mixed": _description(
            "mixed",
            ProcessRole.SETPOINT,
            ProcessRole.CONTROL_SIGNAL,
            ProcessRole.FAULT,
        ),
        "actuator": _description("actuator", ProcessRole.MANIPULATED_VARIABLE),
        "plant": _description("plant", ProcessRole.CONTROLLED_VARIABLE),
        "generic": _description("generic", ProcessRole.PROCESS_VARIABLE),
        "unclassified": _description("unclassified"),
    }

    decisions = _resolve(
        descriptions,
        ("actuator", "plant", "generic", "unclassified"),
        ("mixed",),
    )
    pairs = {decision.relation.signals for decision in decisions}

    assert pairs == {("mixed", "actuator"), ("mixed", "plant"), ("mixed", "generic")}


def test_unknown_target_association_is_not_excluded_but_known_disallowed_class_is():
    descriptions = {
        "controller": _description("controller", ProcessRole.CONTROL_SIGNAL),
        "plant": _description("plant", ProcessRole.PROCESS_VARIABLE),
        "mixed_target": _description(
            "mixed_target",
            ProcessRole.CONTROLLED_VARIABLE,
            ProcessRole.PROCESS_VARIABLE,
        ),
        "unclassified": _description("unclassified"),
    }

    decisions = _resolve(
        descriptions,
        ("plant", "mixed_target", "unclassified", "missing_description"),
        ("controller",),
    )

    assert [
        decision.relation.signals
        for decision in decisions
        if isinstance(decision.relation, ProcessRelation)
    ] == [("controller", "plant")]


def test_composition_preserves_compatible_decisions_and_rejects_conflicts():
    exclusion = _pair_decision("error", "plant", "exclude_predictor")
    compatible = _pair_decision("error", "plant", "exclude_predictor")
    keep = _pair_decision("error", "plant", "keep")
    unrelated = _pair_decision("error", "actuator", "keep")

    assert compose_policy_decisions((exclusion,), (compatible,)) == (exclusion, compatible)
    assert compose_policy_decisions((exclusion,), (unrelated,)) == (exclusion, unrelated)
    with pytest.raises(PolicyDecisionConflict, match="Conflicting policy actions"):
        compose_policy_decisions((exclusion,), (keep,))
