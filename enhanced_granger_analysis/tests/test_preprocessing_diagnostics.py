"""Fit-scoped facts and explicit representation reduction."""

import pandas as pd
import pytest

from ..core.data import (
    ProcessRole,
    RemovedVariable,
    ValueType,
    VariableDescription,
    VariableRoles,
)
from ..preprocessing.contracts import RepresentationAction, RepresentationDecision
from ..preprocessing.diagnostics import (
    ControlErrorDifferenceSpec,
    SignalDiagnostics,
    apply_representation_decision,
)
from ..preprocessing.splitting.scope import FitScope, FitSegment


def _fact(report, kind, variable=None):
    return next(
        fact for fact in report.facts
        if fact.kind == kind and (variable is None or variable in fact.variables)
    )


def test_constants_and_exact_duplicates_are_facts_without_mutating_input():
    source = pd.DataFrame({
        "constant": pd.Series([7, 7, 7], dtype="Int64"),
        "signal": pd.Series(["open", "closed", "open"], dtype="string"),
        "signal_copy": pd.Series(["open", "closed", "open"], dtype="string"),
        "effect": [1.0, 2.0, 3.0],
    })
    original = source.copy(deep=True)
    roles = VariableRoles(
        effects=("effect",),
        causes=("constant", "signal", "signal_copy"),
        tested_causes=("signal_copy",),
    )
    report = SignalDiagnostics(roles).diagnose(
        source, scope=FitScope.all_data([source])
    )

    assert _fact(report, "constant", "constant").variables == ("constant",)
    duplicate = _fact(report, "duplicate", "signal_copy")
    assert duplicate.variables == ("signal_copy", "signal")
    assert duplicate.evidence["retained_variable"] == "signal"
    pd.testing.assert_frame_equal(source, original)
    assert roles.causes == ("constant", "signal", "signal_copy")


def test_constant_detection_keeps_typed_values_and_uses_fit_scope():
    frame = pd.DataFrame({
        "in_scope_constant": pd.Series([1, 1, 2], dtype="Int64"),
        "mixed": pd.Series([1, "1", 1], dtype=object),
        "effect": [0.0, 1.0, 2.0],
    })
    roles = VariableRoles(("effect",), ("in_scope_constant", "mixed"), ())
    scope = FitScope((FitSegment(0, 0, 2),))
    report = SignalDiagnostics(roles).diagnose([frame], scope=scope)

    assert _fact(report, "constant", "in_scope_constant")
    assert not any(
        fact.kind == "constant" and "mixed" in fact.variables
        for fact in report.facts
    )


def _descriptions():
    return {
        "error": VariableDescription("error", ValueType.CONTINUOUS, (ProcessRole.CONTROL_ERROR,)),
        "setpoint": VariableDescription("setpoint", ValueType.CONTINUOUS, (ProcessRole.SETPOINT,)),
        "measured": VariableDescription(
            "measured", ValueType.CONTINUOUS, (ProcessRole.CONTROLLED_VARIABLE,)
        ),
        "alternate": VariableDescription(
            "alternate", ValueType.CONTINUOUS, (ProcessRole.PROCESS_VARIABLE,)
        ),
        "baseline": VariableDescription(
            "baseline", ValueType.CONTINUOUS, (ProcessRole.PROCESS_VARIABLE,)
        ),
    }


def _difference_detector(frame, pairs, *, atol=1e-8, rtol=0.0):
    roles = VariableRoles(
        effects=("error",),
        causes=("setpoint", "measured", "alternate", "baseline"),
        tested_causes=(),
    )
    config = ControlErrorDifferenceSpec(pairs, atol=atol, rtol=rtol)
    return SignalDiagnostics(
        roles,
        variable_descriptions=_descriptions(),
        control_error_difference=config,
    ).diagnose(frame, scope=FitScope.all_data([frame]))


def test_control_error_difference_uses_strict_residual_tolerance():
    frame = pd.DataFrame({
        "error": [2.0 + 1e-9, 3.0 - 1e-9, 1.0],
        "setpoint": [4.0, 5.0, 3.0],
        "measured": [2.0, 2.0, 2.0],
        "alternate": [6.0, 7.0, 5.0],
        "baseline": [4.0, 4.0, 4.0],
    })
    report = _difference_detector(frame, {"error": (("setpoint", "measured"),)})
    fact = _fact(report, "control_error_difference", "error")

    assert fact.evidence["status"] == "supported"
    assert fact.evidence["matches"][0]["match_kind"] == "within_numeric_tolerance"
    relation = fact.evidence["relation"]
    assert relation.target == "error"
    assert [(term.signal, term.coefficient) for term in relation.terms] == [
        ("setpoint", 1.0),
        ("measured", -1.0),
    ]


def test_control_error_difference_reports_no_match_not_linear_prediction():
    frame = pd.DataFrame({
        "error": [2.0, 3.5, 1.0],
        "setpoint": [4.0, 5.0, 3.0],
        "measured": [2.0, 2.0, 2.0],
        "alternate": [6.0, 7.0, 5.0],
        "baseline": [4.0, 4.0, 4.0],
    })
    report = _difference_detector(frame, {"error": (("setpoint", "measured"),)})
    fact = _fact(report, "control_error_difference", "error")

    assert fact.evidence["status"] == "no_match"
    assert fact.evidence["matches"] == ()
    assert fact.evidence["considered"][0]["status"] == "outside_tolerance"


def test_difference_check_does_not_coerce_nonnumeric_values():
    frame = pd.DataFrame({
        "error": ["high", "low"],
        "setpoint": ["high", "low"],
        "measured": ["low", "low"],
        "alternate": ["x", "y"],
        "baseline": ["a", "a"],
    })
    report = _difference_detector(frame, {"error": (("setpoint", "measured"),)})
    fact = _fact(report, "control_error_difference", "error")

    assert fact.evidence["status"] == "no_match"
    assert fact.evidence["considered"][0]["status"] == "non_numeric_or_non_finite"


def test_control_error_difference_ambiguity_is_reported_deterministically():
    frame = pd.DataFrame({
        "error": [2.0, 3.0, 1.0],
        "setpoint": [4.0, 5.0, 3.0],
        "measured": [2.0, 2.0, 2.0],
        "alternate": [6.0, 7.0, 5.0],
        "baseline": [4.0, 4.0, 4.0],
    })
    pairs = (
        ("alternate", "baseline"),
        ("setpoint", "measured"),
    )
    first = _fact(
        _difference_detector(frame, {"error": pairs}),
        "control_error_difference",
        "error",
    )
    second = _fact(
        _difference_detector(frame, {"error": tuple(reversed(pairs))}),
        "control_error_difference",
        "error",
    )

    assert first.evidence["status"] == "ambiguous"
    assert [match["pair"] for match in first.evidence["matches"]] == [
        ("alternate", "baseline"),
        ("setpoint", "measured"),
    ]
    assert first.evidence["matches"] == second.evidence["matches"]
    assert first.evidence["relation"] is None


def test_fixed_decision_applies_to_new_data_without_redetection():
    training = pd.DataFrame({
        "error": [2.0, 3.0],
        "setpoint": [4.0, 5.0],
        "measured": [2.0, 2.0],
        "alternate": [6.0, 7.0],
        "baseline": [4.0, 4.0],
    })
    roles = VariableRoles(
        effects=("error",),
        causes=("error", "setpoint", "measured"),
        tested_causes=("error",),
    )
    descriptions = _descriptions()
    report = SignalDiagnostics(
        roles,
        variable_descriptions=descriptions,
        control_error_difference=ControlErrorDifferenceSpec(
            {"error": (("setpoint", "measured"),)}
        ),
    ).diagnose(training, scope=FitScope.all_data([training]))
    fact = _fact(report, "control_error_difference", "error")
    decision = RepresentationDecision((
        RepresentationAction(
            "error",
            "remove_variable",
            fact_ids=(fact.fact_id,),
            source="explicit-test-policy",
        ),
    ))

    new_data = training.copy(deep=True)
    new_data.loc[:, "error"] = [900.0, -900.0]
    original = new_data.copy(deep=True)
    applied = apply_representation_decision([new_data], roles, report, decision)

    assert applied.effective_roles == VariableRoles(
        effects=(),
        causes=("setpoint", "measured"),
        tested_causes=(),
    )
    assert "error" not in applied.frames[0].columns
    assert "error" in new_data.columns
    pd.testing.assert_frame_equal(new_data, original)
    removal = applied.removals[0]
    assert removal.removal == RemovedVariable(
        "error",
        "algebraic_redundancy",
        ("effect", "cause", "tested_cause"),
    )
    assert removal.defining_relation == fact.evidence["relation"]
    assert removal.decision_source == "explicit-test-policy"


def test_reducer_rejects_ambiguous_or_unreferenced_decisions():
    frame = pd.DataFrame({
        "error": [2.0, 3.0],
        "setpoint": [4.0, 5.0],
        "measured": [2.0, 2.0],
        "alternate": [6.0, 7.0],
        "baseline": [4.0, 4.0],
    })
    roles = VariableRoles(("error",), ("error", "setpoint", "measured", "alternate", "baseline"), ())
    report = SignalDiagnostics(
        roles,
        variable_descriptions=_descriptions(),
        control_error_difference=ControlErrorDifferenceSpec({
            "error": (("setpoint", "measured"), ("alternate", "baseline"))
        }),
    ).diagnose(frame, scope=FitScope.all_data([frame]))
    fact = _fact(report, "control_error_difference", "error")
    decision = RepresentationDecision((
        RepresentationAction("error", "remove_variable", fact_ids=(fact.fact_id,)),
    ))

    with pytest.raises(ValueError, match="uniquely supported"):
        apply_representation_decision([frame], roles, report, decision)
    with pytest.raises(ValueError, match="cite diagnostic"):
        apply_representation_decision(
            [frame],
            roles,
            report,
            RepresentationDecision((RepresentationAction("error", "remove_variable"),)),
        )
