import traceback
from importlib.util import find_spec

import numpy as np
import pandas as pd

from ..api.orchestrator import MultiTaskGrangerAPI
from ..core.lag_config import LagConfiguration
from ..preprocessing.stationarity import StationarityTransformer


from unittest import SkipTest


def _require_pytorch_stack() -> None:
    # Attempt actual imports to detect runtime import errors (some torch builds
    # raise during import even when the package is present). Skip the test if
    # import fails for any reason.
    try:
        import importlib
        importlib.import_module("torch")
    except Exception:
        raise SkipTest("PyTorch is not installed or failed to import")

    try:
        import importlib
        importlib.import_module("pandas")
    except Exception:
        raise SkipTest("pandas is not installed or failed to import")


def _make_demo_data(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    x = rng.normal(size=n_rows)
    y = np.zeros(n_rows, dtype=np.float64)

    for t in range(2, n_rows):
        # Lightweight causal structure: x(t-1), x(t-2) influence y(t)
        y[t] = 0.6 * x[t - 1] - 0.25 * x[t - 2] + 0.05 * rng.normal()

    df = pd.DataFrame(np.column_stack([x, y]), dtype=np.float64, columns=pd.Index(("x", "y")))
    df.columns = ("x", "y")
    return df


def test_pytorch_regularization_hyperopt_runs_in_orchestrator():
    _require_pytorch_stack()

    api = MultiTaskGrangerAPI(backend="pytorch")
    data = _make_demo_data()

    output = api.fit(
        data=data,
        causes=["x"],
        effects=["y"],
        tested_causes=["x"],
        lag_config=LagConfiguration(max_lag=2, use_lag_zero=False),
        stationarity_transformer=StationarityTransformer(max_differencing_order=0),
        regularizer_spec={"type": "l1", "l1": 1e-3},
        hiperoptimalization_state="regularization",
        hiperoptimalization_conf={
            "param_grid": {
                "l1": [1e-4, 1e-3, 1e-2],
            },
            "n_trials": 3,
        },
        model_config={
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 1e-2,
            "device": "cpu",
            "verbose": 0,
        },
    )

    assert output is not None
    assert output.results is not None
    assert "x" in output.reference_models
    assert output.results.p_value.shape == (1, 1)


def test_pytorch_model_hyperopt_path_runs_in_orchestrator():
    _require_pytorch_stack()

    api = MultiTaskGrangerAPI(backend="pytorch")
    data = _make_demo_data()

    output = api.fit(
        data=data,
        causes=["x"],
        effects=["y"],
        tested_causes=["x"],
        lag_config=LagConfiguration(max_lag=2, use_lag_zero=False),
        stationarity_transformer=StationarityTransformer(max_differencing_order=0),
        regularizer_spec={"type": "l1", "l1": 1e-3},
        hiperoptimalization_state="model",
        hiperoptimalization_conf={
            # Current model hyperopt implementation is a no-op, but this keeps the path exercised.
            "param_grid": {"alpha": [1e-3, 1e-2]},
            "n_trials": 2,
        },
        model_config={
            "epochs": 5,
            "batch_size": 16,
            "learning_rate": 1e-2,
            "device": "cpu",
            "verbose": 0,
        },
    )

    assert output is not None
    assert output.results is not None
    assert "x" in output.reference_models


if __name__ == "__main__":
    tests = [
        test_pytorch_regularization_hyperopt_runs_in_orchestrator,
        test_pytorch_model_hyperopt_path_runs_in_orchestrator,
    ]

    print("\n" + "=" * 80)
    print("PYTORCH BACKEND HYPEROPT TESTS")
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
            traceback.print_exc(limit=1)
            failed += 1

    total = len(tests)
    print("-" * 80)
    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed, {skipped}/{total} skipped")
    print("=" * 80 + "\n")
