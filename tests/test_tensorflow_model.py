import os
import sys
import traceback
from pathlib import Path
from importlib.util import find_spec

import numpy as np

# Keras backend selection should happen before TensorFlow/Keras imports.
os.environ["KERAS_BACKEND"] = "tensorflow"

# Allow running this file directly from its nested location, e.g.:
# python complex_granger_analysis/tests/test_tensorflow_model.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complex_granger_analysis.core.exceptions import TrainingError
from complex_granger_analysis.backends.models.tensorflow_model import TensorFlowGrangerModel
from complex_granger_analysis.backends.regularizers.tensorflow_regularizers import (
    KerasL1Regularizer,
    KerasLagDependentL1Regularizer,
)
from complex_granger_analysis.backends.constraints import (
    build_tensorflow_constraint_from_relations,
    process_user_relations,
)


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


def _require_tensorflow():
    if find_spec("tensorflow") is None:
        raise SkipTest("TensorFlow is not installed")


def test_tensorflow_model_initialize_fit_and_omit_variables():
    _require_tensorflow()

    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 5)).astype(np.float64)
    W = rng.normal(size=(5, 2)).astype(np.float64)
    y = X @ W

    model = TensorFlowGrangerModel(
        optimizer="adam",
        loss="mse",
        epochs=5,
        batch_size=10,
        verbose=0,
    )
    model.initialize(X, lags=2, targets=y)

    model.omit_variables([0, 3])
    diag_after = np.diag(model._variable_control_layer.get_weights()[0])  # pylint: disable=protected-access
    assert diag_after[0] == 0.0
    assert diag_after[3] == 0.0
    assert diag_after[1] == 1.0

    result = model.fit()
    assert "weights" in result
    assert "forecasts" in result
    assert "history" in result
    assert result["forecasts"].shape == y.shape
    assert len(result["weights"]) == 1


def test_tensorflow_model_initialize_requires_targets():
    _require_tensorflow()

    X = np.random.randn(10, 4)
    model = TensorFlowGrangerModel(epochs=1, batch_size=4, verbose=0)

    _assert_raises(TrainingError, model.initialize, X, 1)


def test_tensorflow_model_hyperoptimize_returns_message():
    _require_tensorflow()

    model = TensorFlowGrangerModel(epochs=1, batch_size=4, verbose=0)
    result = model.hyperoptimize({"alpha": [0.1, 1.0]}, n_trials=3)

    assert isinstance(result, dict)
    assert "message" in result
    assert "nie posiada parametrów do hiperoptymalizacji" in result["message"]


def test_tensorflow_model_accepts_keras_callbacks():
    _require_tensorflow()

    import tensorflow as tf

    class EpochCounter(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_end_count = 0

        def on_epoch_end(self, epoch, logs=None):
            del epoch, logs
            self.epoch_end_count += 1

    counter = EpochCounter()

    rng = np.random.default_rng(55)
    X = rng.normal(size=(24, 3)).astype(np.float64)
    y = rng.normal(size=(24, 1)).astype(np.float64)

    model = TensorFlowGrangerModel(
        optimizer="adam",
        loss="mse",
        callbacks=[counter],
        epochs=3,
        batch_size=8,
        verbose=0,
    )
    model.initialize(X, lags=1, targets=y)
    model.fit()

    assert counter.epoch_end_count == 3


def test_tensorflow_model_accepts_keras_l1_regularizer():
    _require_tensorflow()

    rng = np.random.default_rng(77)
    X = rng.normal(size=(30, 4)).astype(np.float64)
    y = rng.normal(size=(30, 1)).astype(np.float64)

    regularizer = KerasL1Regularizer(l1=1e-3)
    model = TensorFlowGrangerModel(
        regularizer=regularizer,
        optimizer="adam",
        loss="mse",
        epochs=2,
        batch_size=10,
        verbose=0,
    )
    model.initialize(X, lags=1, targets=y)
    result = model.fit()

    assert "history" in result
    assert regularizer.get_params()["l1"] == 1e-3


def test_tensorflow_model_accepts_optimizer_instance_and_resets_between_fit_calls():
    _require_tensorflow()

    import tensorflow as tf

    rng = np.random.default_rng(79)
    X = rng.normal(size=(30, 4)).astype(np.float64)
    y = rng.normal(size=(30, 1)).astype(np.float64)

    opt_instance = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model = TensorFlowGrangerModel(
        optimizer=opt_instance,
        loss="mse",
        epochs=2,
        batch_size=10,
        verbose=0,
    )
    model.initialize(X, lags=1, targets=y)

    model.fit()
    opt = model.model.optimizer  # pylint: disable=protected-access
    assert opt is not None
    iter_after_first = int(opt.iterations.numpy())
    assert iter_after_first > 0

    model.fit()
    iter_after_second = int(model.model.optimizer.iterations.numpy())  # pylint: disable=protected-access

    # Optimizer should be reset before each fit, so final iterations should
    # match one training run (not accumulate across runs).
    assert iter_after_second == iter_after_first


def test_tensorflow_model_accepts_optimizer_class():
    _require_tensorflow()

    import tensorflow as tf

    rng = np.random.default_rng(81)
    X = rng.normal(size=(24, 3)).astype(np.float64)
    y = rng.normal(size=(24, 1)).astype(np.float64)

    model = TensorFlowGrangerModel(
        optimizer=tf.keras.optimizers.SGD,
        loss="mse",
        epochs=2,
        batch_size=8,
        verbose=0,
    )
    model.initialize(X, lags=1, targets=y)
    out = model.fit()

    assert "history" in out


def test_tensorflow_model_accepts_lag_dependent_l1_regularizer():
    _require_tensorflow()

    rng = np.random.default_rng(88)
    X = rng.normal(size=(36, 6)).astype(np.float64)
    y = rng.normal(size=(36, 1)).astype(np.float64)

    # Two predictor blocks with 3 lag columns each.
    regularizer = KerasLagDependentL1Regularizer(
        l1=1e-3,
        lag_weights=[0.0, 1.0, 2.0, 3.0],
        max_lags_per_pred=[3, 3],
        col_offsets=[0, 3],
    )
    model = TensorFlowGrangerModel(
        regularizer=regularizer,
        optimizer="adam",
        loss="mse",
        epochs=2,
        batch_size=12,
        verbose=0,
    )
    model.initialize(X, lags=3, targets=y)
    result = model.fit()

    assert "history" in result
    assert regularizer.get_params()["lag_weights"] == [0.0, 1.0, 2.0, 3.0]


def test_keras_lag_dependent_regularizer_set_lag_layout_after_init():
    _require_tensorflow()

    regularizer = KerasLagDependentL1Regularizer(l1=1e-3)
    regularizer.set_lag_layout(max_lags_per_pred=[5, 5], col_offsets=[0, 5])
    params = regularizer.get_params()

    assert params["max_lags_per_pred"] == [5, 5]
    assert params["col_offsets"] == [0, 5]
    assert len(params["lag_weights"]) == 20


def test_tensorflow_constraint_processes_user_relations_and_mask():
    spec = process_user_relations(
        relations={
            ("y1", "x1"): 0,
            ("y1", "x2"): {"min_abs_sum": 1.5},
        },
        predictor_names=["x1", "x2"],
        output_names=["y1"],
        col_offsets=[0, 3],
        n_features=6,
        base_mask=np.ones((1, 6), dtype=np.float64),
    )

    # Relation y1<-x1 is forced to zero.
    np.testing.assert_allclose(spec.mask[0, 0:3], 0.0)
    # Relation y1<-x2 remains active and has one min-abs-sum rule.
    np.testing.assert_allclose(spec.mask[0, 3:6], 1.0)
    assert len(spec.rules) == 1
    assert spec.rules[0].output_index == 0
    assert spec.rules[0].feature_indices == (3, 4, 5)
    assert np.isclose(spec.rules[0].min_abs_sum, 1.5)


def test_tensorflow_constraint_enforces_zero_mask_and_min_abs_sum():
    _require_tensorflow()

    import tensorflow as tf

    constraint = build_tensorflow_constraint_from_relations(
        relations={
            ("y1", "x1"): 0,
            ("y1", "x2"): 1.5,
        },
        predictor_names=["x1", "x2"],
        output_names=["y1"],
        col_offsets=[0, 3],
        n_features=6,
        base_mask=np.ones((1, 6), dtype=np.float64),
    )

    # Kernel shape in Keras Dense: (n_features, n_outputs) = (6, 1).
    w = tf.convert_to_tensor(np.zeros((6, 1), dtype=np.float64))
    constrained = constraint(w).numpy()

    # First block (x1) must stay zero due to mask.
    np.testing.assert_allclose(constrained[0:3, 0], 0.0)

    # Second block (x2) must satisfy min abs sum >= 1.5.
    sum_abs_x2 = float(np.sum(np.abs(constrained[3:6, 0])))
    assert sum_abs_x2 >= 1.5 - 1e-9
    assert constraint.is_satisfied(tf.convert_to_tensor(constrained, dtype=tf.float64))


def test_tensorflow_constraint_rejects_invalid_kernel_shape():
    _require_tensorflow()

    import tensorflow as tf
    from complex_granger_analysis.core.exceptions import ConstraintConfigurationError

    constraint = build_tensorflow_constraint_from_relations(
        relations={
            ("y1", "x1"): 0,
        },
        predictor_names=["x1", "x2"],
        output_names=["y1"],
        col_offsets=[0, 3],
        n_features=6,
        base_mask=np.ones((1, 6), dtype=np.float64),
    )

    # Correct Dense kernel shape is (n_features, n_outputs) => (6, 1).
    wrong_shape = tf.zeros((1, 6), dtype=tf.float64)
    try:
        _ = constraint(wrong_shape)
        raise AssertionError("Expected ConstraintConfigurationError for invalid kernel shape")
    except ConstraintConfigurationError:
        pass


if __name__ == "__main__":
    tests = [
        test_tensorflow_model_initialize_fit_and_omit_variables,
        test_tensorflow_model_initialize_requires_targets,
        test_tensorflow_model_hyperoptimize_returns_message,
        test_tensorflow_model_accepts_keras_callbacks,
        test_tensorflow_model_accepts_keras_l1_regularizer,
        test_tensorflow_model_accepts_optimizer_instance_and_resets_between_fit_calls,
        test_tensorflow_model_accepts_optimizer_class,
        test_tensorflow_model_accepts_lag_dependent_l1_regularizer,
        test_keras_lag_dependent_regularizer_set_lag_layout_after_init,
        test_tensorflow_constraint_processes_user_relations_and_mask,
        test_tensorflow_constraint_enforces_zero_mask_and_min_abs_sum,
        test_tensorflow_constraint_rejects_invalid_kernel_shape,
    ]

    print("\n" + "=" * 80)
    print("TENSORFLOW MODEL TESTS")
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
