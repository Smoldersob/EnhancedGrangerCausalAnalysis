from __future__ import annotations

from unittest.mock import Mock

import pytest

from ..api.orchestrator import MultiTaskGrangerAPI
from ..configuration.analysis_spec import AnalysisSpec
from ..configuration.model_search_spec import ModelSearchSpec, SearchBudget
from ..modeling.search_configuration import apply_modeling_candidate
from ..hiperopt.space import SearchSpaceParser
from ..hiperopt.strategies import resolve_search_strategy
from ..hiperopt.space.base import CandidateUpdate, SearchCandidate


def _search_declaration():
    return {
        "space": {
            "model_config.batch_size": {"choices": [16, 32]},
            "model_config.learning_rate": {
                "type": "float",
                "low": 0.001,
                "high": 0.01,
            },
        },
        "strategy": {"type": "grid"},
        "objective": {"type": "mse"},
        "budget": {"max_trials": 3, "epochs_per_trial": 5},
        "preparation": {"splitter": None, "random_state": 7},
    }


def test_search_spec_parses_budget_and_preserves_preparation_overrides():
    spec = ModelSearchSpec.from_mapping(_search_declaration())

    assert spec.budget == SearchBudget(max_trials=3, epochs_per_trial=5)
    assert spec.preparation == {
        "splitter": None,
        "random_state": 7,
    }


def test_search_space_grid_generates_expected_candidates():
    space = SearchSpaceParser().parse({
        "model_config.batch_size": {"choices": [16, 32]},
        "model_config.epochs": {"type": "int", "low": 1, "high": 2},
    })
    candidates = list(resolve_search_strategy("grid").candidates(space))

    assert len(candidates) == 4
    assert [
        tuple(update.value for update in candidate.updates)
        for candidate in candidates
    ] == [
        (16, 1),
        (16, 2),
        (32, 1),
        (32, 2),
    ]


def test_modeling_candidate_application_does_not_mutate_baseline():
    baseline = AnalysisSpec.from_mapping({
        "backend": "numpy",
        "model_config": {
            "model": "conditional_linear",
            "optimizer": {"type": "adam", "params": {}},
        },
        "regularizer_spec": {"type": "l1", "params": {}},
    }).modeling
    candidate = SearchCandidate((
        CandidateUpdate(("model_config", "batch_size"), 32),
        CandidateUpdate(("regularizer_spec", "type"), "l2"),
    ))

    selected = apply_modeling_candidate(baseline, candidate)

    assert selected.model_config["batch_size"] == 32
    assert selected.regularizer_spec["type"] == "l2"
    assert "batch_size" not in baseline.model_config
    assert baseline.regularizer_spec["type"] == "l1"


def test_api_passes_search_and_prepared_artifacts_to_runner():
    api = MultiTaskGrangerAPI(backend="numpy")
    expected = object()
    runner = Mock()
    runner.run_analysis.return_value = expected
    api._analysis_runner = runner
    search_prepared = object()

    result = api.fit(
        data=object(),
        causes=["x"],
        effects=["y"],
        search=_search_declaration(),
        prepared_data=object(),
        search_prepared_data=search_prepared,
    )

    assert result is expected
    kwargs = runner.run_analysis.call_args.kwargs
    assert kwargs["search_prepared_data"] is search_prepared
    assert kwargs["spec"].search is not None
    assert kwargs["spec"].search.budget.max_trials == 3


def test_search_rejects_unknown_declarations():
    with pytest.raises(ValueError, match="Unknown model search fields"):
        ModelSearchSpec.from_mapping({
            **_search_declaration(),
            "unexpected": True,
        })
