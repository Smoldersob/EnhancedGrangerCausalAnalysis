"""Contracts shared across preparation, modeling, and results."""

import ast
from dataclasses import replace
from importlib.util import resolve_name
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ..core.components.constraints.prepared import PreparedConstraintData, PreparedMinAbsSumRule
from ..core.components.constraints.serialization import constraint_data_from_config, constraint_data_to_config
from ..core.components.regularizers.layout import RegularizationLayout
from ..core.components.regularizers.prepared import PreparedRegularizerData
from ..core.components.regularizers.serialization import deserialize_regularizer_data, serialize_regularizer_data
from ..core.data import (
    AlgebraicRelation, AlgebraicRelationEvidence, AlgebraicTerm, AnalyticalRoles, DataSplit,
    FeatureIdentity, FeatureLayout, LagSegmentInfo, PreparedData, ProcessRelation, ProcessRole,
    RelationPolicyDecision, RemovedVariable, ValueType, VariableDescription, VariableRoles,
    identity_target_inverse,
)
from ..core.exceptions import ConstraintConfigurationError, DataValidationError
from ..core.lags import StructuralLagConstraints, validate_lag_spec
from ..core.models.input_spec import ModelInputSpec


def test_feature_identity_is_orthogonal_to_source_type_and_follows_physical_order():
    description = VariableDescription(
        "mode", ValueType.CATEGORICAL, (ProcessRole.SETPOINT, ProcessRole.MANUAL_SETPOINT),
    )
    blocks = {"x": slice(2, 3), "mode": slice(0, 2)}
    layout = FeatureLayout.from_slices(blocks, (1, 1, 2), ("auto", "manual", None))
    assert description.value_type is ValueType.CATEGORICAL
    assert description.process_roles == (ProcessRole.SETPOINT, ProcessRole.MANUAL_SETPOINT)
    assert AnalyticalRoles is VariableRoles
    assert layout.features == (
        FeatureIdentity("mode", 1, "auto"), FeatureIdentity("mode", 1, "manual"),
        FeatureIdentity("x", 2),
    )
    blocks["x"] = slice(0, 1)
    assert layout.cause_slices["x"] == slice(2, 3)
    model = ModelInputSpec(
        3, 1, layout.causes, ("y",), layout.cause_slices, layout.feature_lags, layout.components,
    )
    assert model.feature_layout == layout


@pytest.mark.parametrize("causes, blocks, lags", [
    (("x", "z"), {"z": slice(0, 1), "x": slice(1, 2)}, (1, 1)),
    (("x", "z"), {"x": slice(0, 2), "z": slice(1, 2)}, (1, 2)),
    (("x",), {"x": slice(0, 2)}, (1, 1)),
    (("x",), {"x": slice(0, 1, True)}, (1,)),
    (("x",), {"x": slice(0, 1)}, (True,)),
])
def test_feature_layout_rejects_ambiguous_columns(causes, blocks, lags):
    with pytest.raises((ValueError, TypeError)):
        FeatureLayout(causes, blocks, lags)


def _prepared():
    layout = FeatureLayout(("x", "z"), {"x": slice(0, 2), "z": slice(2, 3)}, (1, 3, 0))
    roles = VariableRoles(("v", "y"), ("x", "z"), ("z",))
    X = np.arange(6).reshape(2, 3)
    y = np.arange(4).reshape(2, 2)
    mask = np.array([[1, 0, 1], [0, 1, 0]])
    indices = np.array([1, 3])
    return PreparedData(
        X, 2 * y, X, y, layout, roles, roles, mask,
        DataSplit("source", "test-splitter", {"training": np.array([0]), "validation": np.array([1])}),
        indices, lambda values: 2 * values,
        segments=(
            LagSegmentInfo(0, 6, 1, slice(0, 2), pd.Index(["t1", "t2"])),
            LagSegmentInfo(0, 6, 1, slice(2, 4), pd.Index(["t4", "t5"])),
        ),
        removed_variables={"a": RemovedVariable("a", "structural_exclusion", ("cause",))},
    )


def test_prepared_data_standalone_output_order_split_and_provenance():
    data = _prepared()
    assert data.effects == ("v", "y")
    assert data.tested_cause_slices == {"z": slice(2, 3)}
    assert data.final_lags == {"v": {"x": (1,), "z": (0,)}, "y": {"x": (3,), "z": ()}}
    assert data.split_snapshot()["index_space"] == "prepared_rows"
    assert data.indices_for("validation").tolist() == [1]
    assert [data.sample_indices[i] for i in range(len(data.X_raw))] == [1, 3]
    labels = [label for segment in data.segments for label in segment.retained_index]
    assert [labels[index] for index in data.sample_indices] == ["t2", "t5"]
    assert data.segments[0].dataset_index == data.segments[1].dataset_index
    _, scaled = data.arrays_for("validation")
    _, raw = data.arrays_for("validation", scaled=False)
    np.testing.assert_array_equal(data.inverse_transform_targets(scaled), raw)
    with pytest.raises(DataValidationError, match="shape"):
        data.inverse_transform_targets(np.arange(2))
    assert not hasattr(data, "lag_engine")
    assert not hasattr(data, "y_scaler")


def test_prepared_data_defensive_copies_and_validates_alignment():
    data = _prepared()
    X = data.X_raw.copy()
    mask = data.base_mask.copy()
    indices = data.sample_indices.copy()
    replacement = replace(data, X_raw=X, base_mask=mask, sample_indices=indices)
    X[:] = -1
    mask[:] = False
    indices[:] = 0
    for name in ("X_raw", "base_mask", "sample_indices"):
        np.testing.assert_array_equal(getattr(data, name), getattr(replacement, name))
        assert not getattr(replacement, name).flags.writeable
    assert replace(data, target_inverse=identity_target_inverse).inverse_transform_targets(data.y_raw).shape == (2, 2)
    with pytest.raises(DataValidationError, match="not aligned"):
        replace(data, X_raw=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="shape"):
        replace(data, base_mask=[[1, 0, 1]])
    with pytest.raises(DataValidationError, match="strictly increasing"):
        replace(data, sample_indices=[2, 1])
    with pytest.raises(DataValidationError, match="outside"):
        replace(data, sample_indices=[0, 4])
    with pytest.raises(DataValidationError, match="contiguous"):
        replace(data, segments=data.segments[1:])
    with pytest.raises(DataValidationError, match="not share"):
        replace(data, split=DataSplit("prepared", None, {"training": [0], "test": [0]}))
    with pytest.raises(DataValidationError, match="self lag 0"):
        replace(data, effective_roles=VariableRoles(("z", "y"), ("x", "z"), ("z",)))
    with pytest.raises(DataValidationError, match="at least one output"):
        replace(data, base_mask=[[1, 0, 0], [0, 1, 0]])


def test_split_and_removal_use_canonical_objects_and_validate_provenance():
    from ..preprocessing.splitting.base import DataSplit as PreviousPath

    assert PreviousPath is DataSplit
    indices = np.array([0, 2])
    split = DataSplit("prepared", None, {"training": indices}).validate(3)
    indices[0] = 1
    assert split.part_names == ("training", "validation", "test")
    assert split.to_dict(n_rows=3)["parts"] == {"training": [0, 2], "validation": [], "test": []}
    with pytest.raises(ValueError, match="one label"):
        LagSegmentInfo(0, 3, 1, slice(0, 2), pd.Index(["too few"]))
    assert RemovedVariable("a", "duplicate", ["cause"], "b").removed_roles == ("cause",)
    with pytest.raises(ValueError, match="Only duplicate"):
        RemovedVariable("a", "no_selected_lags", ("cause",), "b")


def test_partial_lag_constraints_differ_from_complete_selection():
    raw = {"y": {"x": [], "z": [2, 1]}}
    constraints = StructuralLagConstraints(
        allowed_lags=raw, default_allowed_lags=(4,), global_forbidden_lags=(3,),
        required_lags={"y": {"z": (2,)}}, forbidden_lags={"y": {"z": (1,)}},
    )
    raw["y"]["x"].append(99)
    assert constraints.allowed_lags["y"]["x"] == ()
    assert "v" not in constraints.allowed_lags["y"]
    assert constraints.allowed_lags["y"]["z"] == (1, 2)
    with pytest.raises(TypeError):
        constraints.allowed_lags["y"]["x"] = (99,)
    with pytest.raises(ValueError, match="required lag 2 is forbidden"):
        replace(constraints, global_forbidden_lags=(2,))
    with pytest.raises(ValueError, match="required lag 1 is not allowed"):
        replace(constraints, required_lags={"y": {"x": (1,)}})
    assert validate_lag_spec(
        {"y": {"z": (3, 1), "x": ()}}, outputs=("y",), predictors=("x", "z"),
    ) == {"y": {"x": (), "z": (1, 3)}}
    with pytest.raises(ValueError, match="exactly the specified predictors"):
        validate_lag_spec({"y": {"x": ()}}, outputs=("y",), predictors=("x", "z"))
    with pytest.raises(ValueError, match="Self lag 0"):
        validate_lag_spec({"y": {"y": [0]}}, outputs=("y",), predictors=("y",))


def test_relation_evidence_and_decision_have_no_implicit_policy():
    relation = AlgebraicRelation("error", (AlgebraicTerm("setpoint", 1), AlgebraicTerm("measured", -1)))
    with pytest.raises(ValueError, match="finite"):
        AlgebraicTerm("x", float("inf"))
    with pytest.raises(ValueError, match="distinct"):
        AlgebraicRelation("y", (AlgebraicTerm("x", 1), AlgebraicTerm("x", 2)))
    with pytest.raises(ValueError, match="contemporaneous target"):
        AlgebraicRelation("x", (AlgebraicTerm("x", 1),))
    evidence = AlgebraicRelationEvidence(relation, "check-v1", "verification", "contradicted")
    assert not hasattr(evidence, "action")
    assert not hasattr(relation, "status")
    process = ProcessRelation("loop-1", "feedback-association", ("error", "measured"))
    decision = RelationPolicyDecision(process, "plant-policy", "exclude_predictor", "explicit rule", predictor="error")
    assert decision.strength == "suggestion"
    with pytest.raises(ValueError, match="participating"):
        replace(decision, predictor="unknown")
    hard = RelationPolicyDecision(
        relation, "plant-policy", "restrict_lags", "explicit rule", strength="hard",
        lag_constraints=StructuralLagConstraints(allowed_lags={"measured": {"error": ()}}),
    )
    assert hard.lag_constraints.allowed_lags["measured"]["error"] == ()


def test_regularizer_versions_and_constraint_data_round_trip():
    old = {
        "schema_version": 1,
        "layout": {
            "cause_slices": {"a": [1, 2], "z": [0, 1]},
            "output_names": ["out"], "feature_lags": [1, 2], "base_mask": [[True, False]],
        },
        "params": {},
    }
    restored = deserialize_regularizer_data(json.loads(json.dumps(old, sort_keys=True)))
    assert restored.layout.feature_layout.causes == ("z", "a")
    assert serialize_regularizer_data(restored) == old
    layout = RegularizationLayout(
        {"z": slice(0, 2)}, ("out",), [1, 1], [[1, 1]], ("a", "b"),
    )
    payload = serialize_regularizer_data(PreparedRegularizerData(layout))
    assert payload["schema_version"] == 2
    assert deserialize_regularizer_data(json.loads(json.dumps(payload))).layout.feature_layout == layout.feature_layout
    value = PreparedConstraintData(
        ("out",), np.int64(2), [[True, False]],
        (PreparedMinAbsSumRule(np.int64(0), (np.int64(0),), np.float32(0.5)),),
    )
    snap = json.loads(json.dumps(constraint_data_to_config(value)))
    assert constraint_data_to_config(constraint_data_from_config(snap)) == snap
    with pytest.raises(ConstraintConfigurationError, match="only 0 and 1"):
        constraint_data_from_config(dict(snap, mask=[[0.5, 0]]))


def test_core_sources_and_fresh_imports_do_not_depend_on_executors():
    package_path = Path(__file__).resolve().parents[1]
    forbidden_modules = {
        f"{package_path.name}.{name}"
        for name in ("preprocessing", "backends", "modeling", "execution", "api")
    }
    for path in (package_path / "core").rglob("*.py"):
        package = ".".join((package_path.name, *path.relative_to(package_path).parts[:-1]))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = resolve_name("." * node.level + base, package)
                targets = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            else:
                continue
            for target in targets:
                assert not any(
                    target == prefix or target.startswith(prefix + ".") for prefix in forbidden_modules
                ), (path, node.lineno, target)

    subprocess.run(
        [sys.executable, "-c", f"""
import sys
from {package_path.name}.core.data import PreparedData
from {package_path.name}.core.lags import StructuralLagConstraints
from {package_path.name}.core.components.constraints.serialization import constraint_data_to_config
from {package_path.name}.core.components.regularizers.serialization import serialize_regularizer_data
prefixes = {tuple(sorted(forbidden_modules))!r} + ('tensorflow', 'torch', 'keras')
assert not [name for name in sys.modules if any(
    name == prefix or name.startswith(prefix + '.') for prefix in prefixes
)]
"""],
        cwd=package_path.parent, check=True,
    )
