import sys
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.api.builder import MultitaskGrangerBuilder
from complex_granger_analysis.core.exceptions import DataValidationError
import complex_granger_analysis.api.builder as builder_module


class SkipTest(Exception):
    pass


def test_builder_requires_data_before_fit():
    b = MultitaskGrangerBuilder()
    try:
        b.fit()
        assert False, "fit() should fail when data is missing"
    except DataValidationError as exc:
        assert "requires data" in str(exc)


def test_builder_from_config_and_fluent_forward_to_orchestrator_fit():
    captured = {}

    class DummyAPI:
        def __init__(self, backend=None):
            captured["backend"] = backend

        def fit(self, data, **kwargs):
            captured["data"] = data
            captured["kwargs"] = kwargs
            return {"ok": True}

    old_api = builder_module.MultiTaskGrangerAPI
    builder_module.MultiTaskGrangerAPI = DummyAPI
    try:
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [0.5, 0.7, 0.8, 1.0],
            }
        )

        out = (
            MultitaskGrangerBuilder(backend="pytorch")
            .from_config(
                {
                    "x_scaler": "standard",
                    "y_scaler": "standard",
                    "model_config": {"epochs": 10},
                }
            )
            .data(df)
            .variables(causes=["x1"], effects=["x2"], tested_causes=["x1"])
            .backend_load(backend_sample_fraction=0.5, backend_max_samples=2)
            .hyperoptimization(state="model", config={"n_trials": 3, "param_grid": {"epochs": [5, 10]}})
            .fit()
        )

        assert isinstance(out, dict)
        assert out.get("ok") is True

        assert captured["backend"] == "pytorch"
        assert captured["data"].equals(df)

        kwargs = captured["kwargs"]
        assert kwargs["causes"] == ["x1"]
        assert kwargs["effects"] == ["x2"]
        assert kwargs["tested_causes"] == ["x1"]
        assert kwargs["backend_sample_fraction"] == 0.5
        assert kwargs["backend_max_samples"] == 2
        assert kwargs["hiperoptimalization_state"] == "model"
        assert kwargs["hiperoptimalization_conf"]["n_trials"] == 3
        assert kwargs["model_config"]["epochs"] == 10
    finally:
        builder_module.MultiTaskGrangerAPI = old_api


if __name__ == "__main__":
    tests = [
        test_builder_requires_data_before_fit,
        test_builder_from_config_and_fluent_forward_to_orchestrator_fit,
    ]

    print("\n" + "=" * 80)
    print("BUILDER API TESTS")
    print("=" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except SkipTest as exc:
            print(f"SKIP: {name} -> {exc}")
            skipped += 1
        except Exception as exc:
            print(f"FAIL: {name} -> {exc}")
            traceback.print_exc(limit=2)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(
        f"Summary: {passed}/{total} passed, {failed}/{total} failed, {skipped}/{total} skipped"
    )
    print("=" * 80 + "\n")
