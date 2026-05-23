import traceback
from importlib.util import find_spec

import numpy as np

from ..backends import BackendFactory
from unittest import SkipTest


def test_backend_factory_lists_available_backends():
    """Verify that BackendFactory reports at least one available backend."""
    BackendFactory.reset_cache()
    available = BackendFactory.list_available_backends()
    print(f"Available backends: {available}")
    assert len(available) > 0, "No backends available; install tensorflow, torch, or scikit-learn"


def test_backend_factory_tensorflow_strategy():
    if find_spec("tensorflow") is None:
        raise SkipTest("TensorFlow is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("tensorflow")

    assert strategy is not None
    assert strategy.is_available()
    # Strategy interface no longer exposes `get_model_hyperparameters`.
    # Verify the strategy can build a model with expected training kwargs instead.
    model = strategy.build_model(
        n_features=4,
        n_outputs=1,
        regularizer=None,
        constraint=None,
        scaler=None,
        epochs=2,
        batch_size=8,
    )
    assert model is not None


def test_backend_factory_pytorch_strategy():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("pytorch")

    assert strategy is not None
    assert strategy.is_available()
    # Verify strategy can build a model accepting standard training kwargs.
    model = strategy.build_model(
        n_features=4,
        n_outputs=1,
        regularizer=None,
        constraint=None,
        scaler=None,
        epochs=2,
        learning_rate=0.01,
    )
    assert model is not None


def test_backend_factory_sklearn_strategy():
    if find_spec("sklearn") is None:
        raise SkipTest("scikit-learn is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("sklearn")

    assert strategy is not None
    assert strategy.is_available()
    # Verify strategy can build a sklearn-compatible model using max_iter.
    model = strategy.build_model(
        n_features=3,
        n_outputs=1,
        regularizer=None,
        constraint=None,
        scaler=None,
        max_iter=10,
    )
    assert model is not None


def test_backend_factory_get_preferred_backend():
    """Verify that default (preferred) backend selection works."""
    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy(None)  # Should return preferred

    assert strategy is not None
    assert strategy.is_available()


def test_backend_factory_build_model_tensorflow():
    if find_spec("tensorflow") is None:
        raise SkipTest("TensorFlow is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("tensorflow")

    model = strategy.build_model(
        n_features=10,
        n_outputs=2,
        regularizer=None,
        constraint=None,
        scaler=None,
        epochs=5,
        batch_size=16,
    )

    assert model is not None
    assert model.get_backend() == "tensorflow"


def test_backend_factory_build_model_tensorflow_with_object_specs():
    if find_spec("tensorflow") is None:
        raise SkipTest("TensorFlow is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("tensorflow")

    model = strategy.build_model(
        n_features=8,
        n_outputs=1,
        regularizer=None,
        constraint=None,
        scaler=None,
        optimizer={"type": "adam", "learning_rate": 0.001},
        callbacks=[{"type": "early_stopping", "monitor": "loss", "patience": 2}],
        epochs=2,
    )

    assert model is not None
    assert model.get_backend() == "tensorflow"
    assert isinstance(model.callbacks, list)
    assert len(model.callbacks) == 1


def test_backend_factory_build_model_pytorch():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("pytorch")

    model = strategy.build_model(
        n_features=10,
        n_outputs=2,
        regularizer=None,
        constraint=None,
        scaler=None,
        epochs=5,
        batch_size=16,
    )

    assert model is not None
    assert model.get_backend() == "pytorch"


def test_backend_factory_build_model_pytorch_with_object_specs():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("pytorch")

    model = strategy.build_model(
        n_features=6,
        n_outputs=2,
        regularizer=None,
        constraint=None,
        scaler=None,
        optimizer={"type": "adam", "weight_decay": 0.0},
        callbacks=[{"type": "early_stopping", "patience": 2}],
        epochs=2,
    )

    assert model is not None
    assert model.get_backend() == "pytorch"
    assert isinstance(model.callbacks, list)
    assert len(model.callbacks) == 1


def test_backend_factory_build_model_sklearn():
    if find_spec("sklearn") is None:
        raise SkipTest("scikit-learn is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("sklearn")

    model = strategy.build_model(
        n_features=10,
        n_outputs=2,
        regularizer=None,
        constraint=None,
        scaler=None,
        max_iter=10,
    )

    assert model is not None
    assert model.get_backend() == "sklearn"


def test_backend_factory_constraint_from_relations_tensorflow():
    if find_spec("tensorflow") is None:
        raise SkipTest("TensorFlow is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("tensorflow")

    constraint = strategy.build_constraint_from_relations(
        relations={
            ("y1", "x1"): 0,
            ("y1", "x2"): 1.5,
        },
        predictor_names=["x1", "x2"],
        output_names=["y1"],
        col_offsets=np.array([0, 3], dtype=int),
        n_features=6,
        base_mask=np.ones((1, 6), dtype=np.float64),
    )

    assert constraint is not None


def test_backend_factory_regularizer_from_spec():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")

    BackendFactory.reset_cache()
    strategy = BackendFactory.get_strategy("pytorch")

    regularizer = strategy.build_regularizer(
        {"type": "L1", "l1": 0.01}
    )

    assert regularizer is not None


def test_backend_factory_invalid_backend():
    BackendFactory.reset_cache()
    try:
        BackendFactory.get_strategy("invalid_backend_name_xyz")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Unknown backend" in str(e)


if __name__ == "__main__":
    tests = [
        test_backend_factory_lists_available_backends,
        test_backend_factory_tensorflow_strategy,
        test_backend_factory_pytorch_strategy,
        test_backend_factory_sklearn_strategy,
        test_backend_factory_get_preferred_backend,
        test_backend_factory_build_model_tensorflow,
        test_backend_factory_build_model_tensorflow_with_object_specs,
        test_backend_factory_build_model_pytorch,
        test_backend_factory_build_model_pytorch_with_object_specs,
        test_backend_factory_build_model_sklearn,
        test_backend_factory_constraint_from_relations_tensorflow,
        test_backend_factory_regularizer_from_spec,
        test_backend_factory_invalid_backend,
    ]

    print("\n" + "=" * 80)
    print("BACKEND FACTORY TESTS")
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
