# enhanced_granger_analysis/tests/test_constraint_configuration.py

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enhanced_granger_analysis.api.builder import MultiTaskGrangerBuilder
from enhanced_granger_analysis.api.config_loader import (
    BuilderConfigLoader,
    TestGroupConfigIterator as GroupConfigIterator,
)
from enhanced_granger_analysis.core.components.constraint_parsing import (
    parse_relations_spec,
)
from enhanced_granger_analysis.core.components.constraint_preparation import (
    prepare_constraint_data,
)
from enhanced_granger_analysis.core.components.constraint_spec import (
    MinAbsSumRelationRule,
    PreparedConstraintData,
    RelationConstraint,
    RelationConstraintsSpec,
    ZeroRelationRule,
)
from enhanced_granger_analysis.core.exceptions import (
    ConstraintConfigurationError,
    DataValidationError,
)


@pytest.fixture
def declarations():
    return [
        {
            "cause": "A",
            "effect": "Y",
            "rule": {"type": "zero"},
        },
        {
            "cause": "B",
            "effect": "Y",
            "rule": {
                "type": "min_abs_sum",
                "params": {"value": 0.2},
            },
        },
    ]


@pytest.fixture
def prepared_data():
    return SimpleNamespace(
        cause_slices={
            "B": slice(2, 5),
            "A": slice(0, 2),
        },
        causes=("A", "B"),
        effects=("Y", "Z"),
        tested_causes=("A",),
        X_scaled=np.zeros((8, 5)),
        base_mask=np.array(
            [
                [1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_canonical_declarations(declarations):
    spec = parse_relations_spec(declarations)

    assert isinstance(spec, RelationConstraintsSpec)
    assert spec.relations == (
        RelationConstraint("A", "Y", ZeroRelationRule()),
        RelationConstraint("B", "Y", MinAbsSumRelationRule(0.2)),
    )


def test_parse_ready_spec_preserves_identity(declarations):
    spec = parse_relations_spec(declarations)

    assert parse_relations_spec(spec) is spec


def test_empty_list_creates_empty_spec():
    assert parse_relations_spec([]) == RelationConstraintsSpec()


def test_canonical_round_trip(declarations):
    spec = parse_relations_spec(declarations)

    assert spec.to_config() == declarations
    assert parse_relations_spec(spec.to_config()) == spec


def test_parser_does_not_mutate_input(declarations):
    original = copy.deepcopy(declarations)

    parse_relations_spec(declarations)

    assert declarations == original


def test_spec_is_independent_of_input_mutations(declarations):
    spec = parse_relations_spec(declarations)

    declarations[0]["cause"] = "changed"
    declarations[1]["rule"]["params"]["value"] = 99
    declarations.clear()

    assert spec.relations[0].cause == "A"
    assert spec.relations[1].rule.value == 0.2


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"rules": []},
        {"relations": []},
        {("Y", "A"): 0},
        {"Y->A": {"zero": True}},
        "relations.json",
        '[{"cause": "A", "effect": "Y", "rule": {"type": "zero"}}]',
        (),
        RelationConstraintsSpec,
    ],
)
def test_parser_rejects_unsupported_root_formats(raw):
    with pytest.raises(ConstraintConfigurationError):
        parse_relations_spec(raw)


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        "A->Y",
        {},
        {"cause": "A", "effect": "Y"},
        {"cause": "A", "rule": {"type": "zero"}},
        {"effect": "Y", "rule": {"type": "zero"}},
        {
            "predictor": "A",
            "target": "Y",
            "rule": {"type": "zero"},
        },
        {
            "cause": "A",
            "effect": "Y",
            "zero": True,
        },
        {
            "cause": "A",
            "effect": "Y",
            "rule": {"type": "zero"},
            "unexpected": True,
        },
    ],
)
def test_parser_rejects_invalid_declarations(declaration):
    with pytest.raises(ConstraintConfigurationError):
        parse_relations_spec([declaration])


@pytest.mark.parametrize("field", ["cause", "effect"])
@pytest.mark.parametrize("value", [None, "", "   ", 1, True, []])
def test_parser_rejects_invalid_variable_names(field, value):
    declaration = {
        "cause": "A",
        "effect": "Y",
        "rule": {"type": "zero"},
    }
    declaration[field] = value

    with pytest.raises(ConstraintConfigurationError):
        parse_relations_spec([declaration])


def test_parser_preserves_exact_variable_names():
    spec = parse_relations_spec([
        {
            "cause": " A ",
            "effect": "Y:output",
            "rule": {"type": "zero"},
        },
    ])

    assert spec.relations[0].cause == " A "
    assert spec.relations[0].effect == "Y:output"


@pytest.mark.parametrize(
    "rule",
    [
        None,
        0,
        False,
        "zero",
        {},
        {"zero": True},
        {"min_abs_sum": 0.2},
        {"type": None},
        {"type": "ZERO"},
        {"type": " zero "},
        {"type": "unknown"},
        {"type": "zero", "params": None},
        {"type": "zero", "params": []},
        {"type": "zero", "params": {"value": 0}},
        {"type": "zero", "value": 0},
        {"type": "min_abs_sum"},
        {"type": "min_abs_sum", "params": {}},
        {
            "type": "min_abs_sum",
            "params": {"value": 0.2, "extra": 1},
        },
    ],
)
def test_parser_rejects_invalid_rule_formats(rule):
    with pytest.raises(ConstraintConfigurationError):
        parse_relations_spec([
            {"cause": "A", "effect": "Y", "rule": rule},
        ])


@pytest.mark.parametrize(
    "value",
    [-1, float("nan"), float("inf"), -float("inf"), True, False, "0.2", None],
)
def test_parser_rejects_invalid_min_abs_sum_values(value):
    with pytest.raises(ConstraintConfigurationError):
        parse_relations_spec([
            {
                "cause": "A",
                "effect": "Y",
                "rule": {
                    "type": "min_abs_sum",
                    "params": {"value": value},
                },
            },
        ])


def test_zero_rule_accepts_empty_params():
    spec = parse_relations_spec([
        {
            "cause": "A",
            "effect": "Y",
            "rule": {"type": "zero", "params": {}},
        },
    ])

    assert spec.relations[0].rule == ZeroRelationRule()


def test_duplicate_relation_is_rejected():
    declarations = [
        {
            "cause": "A",
            "effect": "Y",
            "rule": {"type": "zero"},
        },
        {
            "cause": "A",
            "effect": "Y",
            "rule": {
                "type": "min_abs_sum",
                "params": {"value": 0.2},
            },
        },
    ]

    with pytest.raises(ConstraintConfigurationError, match="Duplicate"):
        parse_relations_spec(declarations)


# ---------------------------------------------------------------------------
# Constraint preparation
# ---------------------------------------------------------------------------


def test_prepare_uses_explicit_slices_and_active_coefficients(
    declarations,
    prepared_data,
):
    original_mask = prepared_data.base_mask.copy()

    result = prepare_constraint_data(
        parse_relations_spec(declarations),
        cause_slices=prepared_data.cause_slices,
        output_names=prepared_data.effects,
        n_features=5,
        base_mask=prepared_data.base_mask,
    )

    np.testing.assert_array_equal(
        result.mask,
        [
            [0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ],
    )
    np.testing.assert_array_equal(
        prepared_data.base_mask,
        original_mask,
    )

    assert result.output_names == ("Y", "Z")
    assert result.n_features == 5
    assert len(result.rules) == 1
    assert result.rules[0].output_index == 0
    assert result.rules[0].feature_indices == (2, 4)
    assert result.rules[0].min_abs_sum == 0.2
    assert not np.shares_memory(result.mask, prepared_data.base_mask)


def test_prepared_mask_is_read_only(prepared_data):
    result = prepare_constraint_data(
        RelationConstraintsSpec(),
        cause_slices=prepared_data.cause_slices,
        output_names=prepared_data.effects,
        n_features=5,
        base_mask=prepared_data.base_mask,
    )

    with pytest.raises(ValueError):
        result.mask[0, 0] = False


def test_empty_spec_preserves_base_mask(prepared_data):
    result = prepare_constraint_data(
        RelationConstraintsSpec(),
        cause_slices=prepared_data.cause_slices,
        output_names=prepared_data.effects,
        n_features=5,
        base_mask=prepared_data.base_mask,
    )

    np.testing.assert_array_equal(result.mask, prepared_data.base_mask)
    assert result.rules == ()


def test_missing_base_mask_defaults_to_all_allowed():
    result = prepare_constraint_data(
        RelationConstraintsSpec(),
        cause_slices={"A": slice(0, 2)},
        output_names=("Y",),
        n_features=2,
    )

    np.testing.assert_array_equal(result.mask, [[True, True]])


def test_zero_lower_bound_is_not_a_zero_relation():
    spec = parse_relations_spec([
        {
            "cause": "A",
            "effect": "Y",
            "rule": {
                "type": "min_abs_sum",
                "params": {"value": 0},
            },
        },
    ])

    result = prepare_constraint_data(
        spec,
        cause_slices={"A": slice(0, 2)},
        output_names=("Y",),
        n_features=2,
    )

    assert result.mask.all()
    assert result.rules == ()


def test_output_order_controls_rule_output_index():
    spec = parse_relations_spec([
        {
            "cause": "A",
            "effect": "Y",
            "rule": {
                "type": "min_abs_sum",
                "params": {"value": 0.2},
            },
        },
    ])

    result = prepare_constraint_data(
        spec,
        cause_slices={"A": slice(0, 2)},
        output_names=("Z", "Y"),
        n_features=2,
    )

    assert result.rules[0].output_index == 1


@pytest.mark.parametrize(
    ("cause", "effect"),
    [("missing", "Y"), ("A", "missing")],
)
def test_unknown_effective_variables_are_rejected(cause, effect):
    spec = parse_relations_spec([
        {
            "cause": cause,
            "effect": effect,
            "rule": {"type": "zero"},
        },
    ])

    with pytest.raises(ConstraintConfigurationError):
        prepare_constraint_data(
            spec,
            cause_slices={"A": slice(0, 2)},
            output_names=("Y",),
            n_features=2,
        )


@pytest.mark.parametrize(
    "layout",
    [
        {},
        {"A": slice(0, 1)},
        {"A": slice(0, 2), "B": slice(1, 2)},
        {"A": slice(-1, 2)},
        {"A": slice(0, 3)},
        {"A": slice(2, 1)},
        {"A": slice(None, 2)},
        {"A": slice(0, None)},
        {"A": slice(0, 2, 2)},
        {"A": slice(0.0, 2)},
        {"A": slice(False, 2)},
        {"A": (0, 2)},
    ],
)
def test_invalid_feature_layout_is_rejected(layout):
    with pytest.raises(ConstraintConfigurationError):
        prepare_constraint_data(
            RelationConstraintsSpec(),
            cause_slices=layout,
            output_names=("Y",),
            n_features=2,
        )


@pytest.mark.parametrize(
    "mask",
    [
        [[1]],
        [[1, 1], [1, 1]],
        [[1, 0.5]],
        [[1, float("nan")]],
        [[1, float("inf")]],
        [["1", "0"]],
    ],
)
def test_invalid_base_mask_is_rejected(mask):
    with pytest.raises(ConstraintConfigurationError):
        prepare_constraint_data(
            RelationConstraintsSpec(),
            cause_slices={"A": slice(0, 2)},
            output_names=("Y",),
            n_features=2,
            base_mask=np.asarray(mask),
        )


@pytest.mark.parametrize("outputs", [(), ("Y", "Y"), ("",), "Y"])
def test_invalid_output_names_are_rejected(outputs):
    with pytest.raises(ConstraintConfigurationError):
        prepare_constraint_data(
            RelationConstraintsSpec(),
            cause_slices={"A": slice(0, 2)},
            output_names=outputs,
            n_features=2,
        )


def test_preparation_rejects_unparsed_declarations(declarations):
    with pytest.raises(ConstraintConfigurationError):
        prepare_constraint_data(
            declarations,
            cause_slices={"A": slice(0, 2), "B": slice(2, 5)},
            output_names=("Y",),
            n_features=5,
        )


@pytest.mark.parametrize("policy", ["error", "warn"])
def test_infeasible_positive_rule_policy(policy):
    spec = parse_relations_spec([
        {
            "cause": "A",
            "effect": "Y",
            "rule": {
                "type": "min_abs_sum",
                "params": {"value": 0.2},
            },
        },
    ])

    kwargs = {
        "cause_slices": {"A": slice(0, 2)},
        "output_names": ("Y",),
        "n_features": 2,
        "base_mask": np.zeros((1, 2)),
        "infeasible_policy": policy,
    }

    if policy == "error":
        with pytest.raises(
            ConstraintConfigurationError,
            match="no active coefficients",
        ):
            prepare_constraint_data(spec, **kwargs)
    else:
        with pytest.warns(UserWarning, match="no active coefficients"):
            result = prepare_constraint_data(spec, **kwargs)

        assert result.rules == ()
        assert not result.mask.any()


# ---------------------------------------------------------------------------
# JSON/YAML loader
# ---------------------------------------------------------------------------


def test_loader_reads_inline_json_relations(tmp_path, declarations):
    path = write_json(
        tmp_path / "config.json",
        {"relations": declarations},
    )

    config = BuilderConfigLoader.load_file(path)

    assert config["relations"] == parse_relations_spec(declarations)


def test_loader_reads_standalone_relation_list(tmp_path, declarations):
    path = write_json(tmp_path / "relations.json", declarations)

    spec = BuilderConfigLoader.load_relations_file(path)

    assert spec == parse_relations_spec(declarations)


def test_loader_resolves_path_relative_to_config(
    tmp_path,
    monkeypatch,
    declarations,
):
    config_dir = tmp_path / "experiment"
    write_json(config_dir / "rules" / "relations.json", declarations)
    config_path = write_json(
        config_dir / "config.json",
        {"relations": "rules/relations.json"},
    )

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    config = BuilderConfigLoader.load_file(config_path)

    assert config["relations"] == parse_relations_spec(declarations)


def test_loader_accepts_absolute_relation_path(tmp_path, declarations):
    relations_path = write_json(
        tmp_path / "relations.json",
        declarations,
    )
    config_path = write_json(
        tmp_path / "config.json",
        {"relations": str(relations_path.resolve())},
    )

    config = BuilderConfigLoader.load_file(config_path)

    assert config["relations"] == parse_relations_spec(declarations)


def test_loader_reads_referenced_yaml(tmp_path, declarations):
    yaml = pytest.importorskip("yaml")
    relations_path = tmp_path / "relations.yaml"
    relations_path.write_text(
        yaml.safe_dump(declarations),
        encoding="utf-8",
    )
    config_path = write_json(
        tmp_path / "config.json",
        {"relations": "relations.yaml"},
    )

    config = BuilderConfigLoader.load_file(config_path)

    assert config["relations"] == parse_relations_spec(declarations)


def test_loader_reads_inline_yaml(tmp_path, declarations):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"relations": declarations}),
        encoding="utf-8",
    )

    config = BuilderConfigLoader.load_file(path)

    assert config["relations"] == parse_relations_spec(declarations)


@pytest.mark.parametrize("reference", ["", "   ", "missing.json", "."])
def test_loader_rejects_invalid_relation_file_reference(tmp_path, reference):
    config_path = write_json(
        tmp_path / "config.json",
        {"relations": reference},
    )

    with pytest.raises(DataValidationError):
        BuilderConfigLoader.load_file(config_path)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"rules": []},
        {"Y->A": 0},
        [{"cause": "A", "effect": "Y", "zero": True}],
    ],
)
def test_loader_rejects_old_or_invalid_inline_formats(tmp_path, raw):
    path = write_json(tmp_path / "config.json", {"relations": raw})

    with pytest.raises(ConstraintConfigurationError):
        BuilderConfigLoader.load_file(path)


def test_loader_rejects_wrapped_relation_file(tmp_path, declarations):
    path = write_json(
        tmp_path / "relations.json",
        {"rules": declarations},
    )

    with pytest.raises(ConstraintConfigurationError):
        BuilderConfigLoader.load_relations_file(path)


def test_loader_normalization_preserves_ready_spec(declarations):
    spec = parse_relations_spec(declarations)

    config = BuilderConfigLoader.normalize_builder_config({
        "relations": spec,
    })

    assert config["relations"] is spec


def test_group_iterator_resolves_relation_file(tmp_path, declarations):
    write_json(tmp_path / "relations.json", declarations)
    group_path = write_json(
        tmp_path / "group.json",
        {
            "base_config": {
                "relations": "relations.json",
                "model_config": {"epochs": 1},
            },
            "sweep": {
                "param_names": ["model_config.epochs"],
                "cases": [[2], [3]],
            },
        },
    )

    configs = list(GroupConfigIterator.from_file(group_path))

    assert [cfg["model_config"]["epochs"] for cfg in configs] == [2, 3]
    assert all(
        cfg["relations"] == parse_relations_spec(declarations)
        for cfg in configs
    )


# ---------------------------------------------------------------------------
# Builder integration
# ---------------------------------------------------------------------------


def test_builder_accepts_canonical_list(declarations):
    builder = MultiTaskGrangerBuilder()

    returned = builder.relations(declarations)

    assert returned is builder
    assert builder._fit_kwargs["relations"] == parse_relations_spec(
        declarations
    )


def test_builder_accepts_ready_spec(declarations):
    builder = MultiTaskGrangerBuilder()
    spec = parse_relations_spec(declarations)

    builder.relations(spec)

    assert builder._fit_kwargs["relations"] is spec


@pytest.mark.parametrize(
    "raw",
    [None, {}, {"rules": []}, {("Y", "A"): 0}, "relations.json"],
)
def test_builder_rejects_unsupported_relation_inputs(raw):
    builder = MultiTaskGrangerBuilder()

    with pytest.raises(ConstraintConfigurationError):
        builder.relations(raw)


def test_builder_invalid_relations_preserve_previous_state(declarations):
    builder = MultiTaskGrangerBuilder().relations(declarations)
    previous_spec = builder._fit_kwargs["relations"]
    previous_constraint = object()
    builder._constraint = previous_constraint

    with pytest.raises(ConstraintConfigurationError):
        builder.relations({"Y->A": 0})

    assert builder._fit_kwargs["relations"] is previous_spec
    assert builder._constraint is previous_constraint


def test_builder_from_config_parses_relations(declarations):
    builder = MultiTaskGrangerBuilder().from_config({
        "relations": declarations,
    })

    assert builder._fit_kwargs["relations"] == parse_relations_spec(
        declarations
    )


def test_builder_from_file_parses_referenced_relations(
    tmp_path,
    declarations,
):
    write_json(tmp_path / "relations.json", declarations)
    config_path = write_json(
        tmp_path / "config.json",
        {"relations": "relations.json"},
    )

    builder = MultiTaskGrangerBuilder().from_file(config_path)

    assert builder._fit_kwargs["relations"] == parse_relations_spec(
        declarations
    )


def test_builder_prepares_constraint_data_without_backend(
    declarations,
    prepared_data,
):
    builder = MultiTaskGrangerBuilder().relations(declarations)
    builder._prepared_data = prepared_data

    result = builder.prepare_constraint_data()

    assert isinstance(result, PreparedConstraintData)
    assert builder._constraint_data is result
    assert builder._strategy is None
    assert result.rules[0].feature_indices == (2, 4)
    assert result.mask.shape == (2, 5)


def test_builder_requires_prepared_data(declarations):
    builder = MultiTaskGrangerBuilder().relations(declarations)

    with pytest.raises(DataValidationError):
        builder.prepare_constraint_data()


def test_relation_change_invalidates_compiled_data(
    declarations,
    prepared_data,
):
    builder = MultiTaskGrangerBuilder().relations(declarations)
    builder._prepared_data = prepared_data
    builder.prepare_constraint_data()
    builder._constraint = object()
    builder._model = object()

    builder.relations([])

    assert builder._constraint_data is None
    assert builder._constraint is None
    assert builder._model is None
    assert builder._prepared_data is prepared_data
    assert builder._fit_kwargs["relations"] == RelationConstraintsSpec()


@pytest.mark.parametrize(
    ("stage", "should_clear"),
    [
        ("data", True),
        ("components", True),
        ("model", False),
    ],
)

def test_constraint_data_invalidation(stage, should_clear):
    builder = MultiTaskGrangerBuilder()
    marker = object()
    builder._constraint_data = marker

    builder._invalidate_from(stage)

    if should_clear:
        assert builder._constraint_data is None
    else:
        assert builder._constraint_data is marker


def test_builder_recompiles_against_current_base_mask(
    declarations,
    prepared_data,
):
    builder = MultiTaskGrangerBuilder().relations(declarations)
    builder._prepared_data = prepared_data

    first = builder.prepare_constraint_data()
    prepared_data.base_mask[0, 2] = False
    second = builder.prepare_constraint_data()

    assert first.rules[0].feature_indices == (2, 4)
    assert second.rules[0].feature_indices == (4,)
    assert builder._constraint_data is second