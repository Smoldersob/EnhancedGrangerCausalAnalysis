from enhanced_granger_analysis.core.data.variables import (
    ProcessRole,
    ValueType,
    VariableDescription,
)
from enhanced_granger_analysis.core.data.analysis_roles import (
    AnalyticalRoles,
    VariableRoles,
)


def test_value_types_and_process_roles_cover_declared_names():
    assert {value.value for value in ValueType} == {
        "continuous",
        "binary",
        "categorical",
        "ordinal",
        "count",
    }
    assert {role.value for role in ProcessRole} >= {
        "process_variable",
        "terminal_output",
        "redundant_measurement",
        "virtual_measurement",
        "setpoint",
        "manual_setpoint",
        "control_signal",
        "manipulated_variable",
        "control_error",
        "controlled_variable",
        "fault",
        "diagnostic_signal",
        "residual",
        "disturbance",
    }


def test_variable_description_supports_multiple_roles_and_domain_metadata():
    description = VariableDescription(
        name="estimated_temperature",
        value_type="continuous",
        process_roles=("virtual_measurement", "controlled_variable"),
        domain_constraints={"minimum": 0.0},
    )

    assert description.value_type is ValueType.CONTINUOUS
    assert description.process_roles == (
        ProcessRole.VIRTUAL_MEASUREMENT,
        ProcessRole.CONTROLLED_VARIABLE,
    )
    assert description.domain_constraints["minimum"] == 0.0


def test_analytical_roles_remain_independent():
    roles = VariableRoles(
        effects=("y",),
        causes=("u", "y"),
        tested_causes=("u",),
    )

    assert AnalyticalRoles is VariableRoles
    assert roles.tested_causes == ("u",)
