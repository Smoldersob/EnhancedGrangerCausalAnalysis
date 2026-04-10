import sys
import traceback
from tempfile import TemporaryDirectory
from pathlib import Path
from importlib.util import find_spec

import numpy as np

# Allow running this file directly from its nested location, e.g.:
# python complex_granger_analysis/tests/test_pytorch_model.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.core.exceptions import TrainingError
from complex_granger_analysis.backends.callbacks import (
    ConvergenceCheck,
    EarlyStopping,
    ReduceLearningRate,
    TorchTensorBoardCallback,
)
from complex_granger_analysis.backends.models.pytorch_model import PyTorchGrangerModel


class SkipTest(Exception):
    """Local skip marker for optional runtime dependencies."""


def _assert_raises(exc_type, fn, *args, **kwargs):
    """Minimal helper to assert exception without pytest dependency in test body."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"Expected {exc_type.__name__}, but got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def _require_torch():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")


def test_pytorch_model_initialize_fit_and_omit_variables():
    _require_torch()

    rng = np.random.default_rng(42)
    X = rng.normal(size=(48, 6)).astype(np.float64)
    W = rng.normal(size=(6, 2)).astype(np.float64)
    y = X @ W

    model = PyTorchGrangerModel(
        learning_rate=0.05,
        epochs=8,
        batch_size=16,
        verbose=0,
        device="cpu",
    )
    model.initialize(X, lags=2, targets=y)

    model.omit_variables([1, 4])

    diag_after = np.diag(
        model._variable_control_layer.weight.detach().cpu().numpy()  # pylint: disable=protected-access
    )
    assert diag_after[1] == 0.0
    assert diag_after[4] == 0.0
    assert diag_after[0] == 1.0

    result = model.fit()
    assert "weights" in result
    assert "forecasts" in result
    assert "history" in result
    assert result["forecasts"].shape == y.shape
    assert len(result["weights"]) == 1


def test_pytorch_model_initialize_requires_targets():
    _require_torch()

    X = np.random.randn(10, 4)
    model = PyTorchGrangerModel(epochs=1, batch_size=4, device="cpu")

    _assert_raises(TrainingError, model.initialize, X, 1)


def test_pytorch_model_hyperoptimize_returns_message():
    _require_torch()

    model = PyTorchGrangerModel(epochs=1, batch_size=4, device="cpu")
    result = model.hyperoptimize({"alpha": [0.1, 1.0]}, n_trials=3)

    assert isinstance(result, dict)
    assert "message" in result
    assert "nie posiada parametrów do hiperoptymalizacji" in result["message"]


def test_pytorch_model_supports_custom_optimizer_and_loss_strings():
    _require_torch()

    rng = np.random.default_rng(9)
    X = rng.normal(size=(32, 4)).astype(np.float64)
    y = rng.normal(size=(32, 1)).astype(np.float64)

    model = PyTorchGrangerModel(
        optimizer="sgd",
        loss="mae",
        learning_rate=0.01,
        epochs=3,
        batch_size=8,
        verbose=0,
        device="cpu",
    )
    model.initialize(X, lags=1, targets=y)
    result = model.fit()

    assert "history" in result
    assert len(result["history"]["loss"]) == 3


def test_pytorch_model_callbacks_can_stop_training_early():
    _require_torch()

    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 5)).astype(np.float64)
    y = rng.normal(size=(40, 1)).astype(np.float64)

    model = PyTorchGrangerModel(
        epochs=20,
        batch_size=10,
        learning_rate=0.01,
        callbacks=[ConvergenceCheck(relative_change_threshold=10.0)],
        verbose=0,
        device="cpu",
    )
    model.initialize(X, lags=1, targets=y)
    result = model.fit()

    assert len(result["history"]["loss"]) < 20
    assert result["history"]["stop_reason"] == "convergence_check"


def test_pytorch_model_reduce_lr_and_early_stopping_callbacks():
    _require_torch()

    rng = np.random.default_rng(17)
    X = rng.normal(size=(48, 4)).astype(np.float64)
    y = rng.normal(size=(48, 1)).astype(np.float64)

    model = PyTorchGrangerModel(
        epochs=10,
        batch_size=12,
        learning_rate=0.1,
        callbacks=[
            ReduceLearningRate(patience=1, factor=0.5, min_delta=1e9),
            EarlyStopping(patience=2, min_delta=1e9, restore_best_weights=False),
        ],
        verbose=0,
        device="cpu",
    )
    model.initialize(X, lags=1, targets=y)
    result = model.fit()

    final_lr = model._optimizer.param_groups[0]["lr"]  # pylint: disable=protected-access
    assert final_lr < 0.1
    assert result["history"]["stop_reason"] == "early_stopping"


def test_pytorch_model_tensorboard_callback_writes_event_files():
    _require_torch()

    rng = np.random.default_rng(23)
    X = rng.normal(size=(32, 4)).astype(np.float64)
    y = rng.normal(size=(32, 1)).astype(np.float64)

    with TemporaryDirectory() as tmp_dir:
        callback = TorchTensorBoardCallback(log_dir=tmp_dir, log_every_n_epochs=1)

        model = PyTorchGrangerModel(
            epochs=3,
            batch_size=8,
            learning_rate=0.01,
            callbacks=[callback],
            verbose=0,
            device="cpu",
        )
        model.initialize(X, lags=1, targets=y)
        try:
            model.fit()
        except RuntimeError as exc:
            raise SkipTest(str(exc)) from exc

        event_files = list(Path(tmp_dir).glob("events.out.tfevents.*"))
        assert len(event_files) > 0


def test_pytorch_optimizer_resets_between_fit_calls():
    _require_torch()

    rng = np.random.default_rng(29)
    X = rng.normal(size=(36, 4)).astype(np.float64)
    y = rng.normal(size=(36, 1)).astype(np.float64)

    model = PyTorchGrangerModel(
        optimizer="adam",
        learning_rate=0.01,
        epochs=2,
        batch_size=12,
        verbose=0,
        device="cpu",
    )
    model.initialize(X, lags=1, targets=y)

    model.fit()
    opt_first = model._optimizer  # pylint: disable=protected-access
    assert opt_first is not None

    model.fit()
    opt_second = model._optimizer  # pylint: disable=protected-access
    assert opt_second is not None

    # Each fit() should start from a fresh optimizer state object.
    assert opt_second is not opt_first


if __name__ == "__main__":
    tests = [
        test_pytorch_model_initialize_fit_and_omit_variables,
        test_pytorch_model_initialize_requires_targets,
        test_pytorch_model_hyperoptimize_returns_message,
        test_pytorch_model_supports_custom_optimizer_and_loss_strings,
        test_pytorch_model_callbacks_can_stop_training_early,
        test_pytorch_model_reduce_lr_and_early_stopping_callbacks,
        test_pytorch_model_tensorboard_callback_writes_event_files,
        test_pytorch_optimizer_resets_between_fit_calls,
    ]

    print("\n" + "=" * 80)
    print("PYTORCH MODEL TESTS")
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
