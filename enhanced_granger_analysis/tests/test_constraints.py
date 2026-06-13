import traceback
from importlib.util import find_spec

import numpy as np

from ..backends.constraints import (
    build_numpy_constraint_from_relations,
)

if find_spec("torch") is not None:
    from ..backends.constraints import (
        build_pytorch_constraint_from_relations,
    )
else:
    build_pytorch_constraint_from_relations = None

if find_spec("sklearn") is not None:
    from ..backends.models.scikit_model import ScikitConstrainedGrangerModel
else:
    ScikitConstrainedGrangerModel = None


class SkipTest(Exception):
    pass


def _require_torch():
    if find_spec("torch") is None:
        raise SkipTest("PyTorch is not installed")
    if build_pytorch_constraint_from_relations is None:
        raise SkipTest("PyTorch constraints are not available")


def test_numpy_constraint_enforces_mask_and_min_abs_sum():
    constraint = build_numpy_constraint_from_relations(
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

    coef = np.zeros((1, 6), dtype=np.float64)
    constrained = constraint(coef)

    np.testing.assert_allclose(constrained[0, 0:3], 0.0)
    sum_abs = float(np.sum(np.abs(constrained[0, 3:6])))
    assert sum_abs >= 1.5 - 1e-9
    assert constraint.is_satisfied(constrained)


def test_numpy_constraint_is_compatible_with_scikit_model():
    if ScikitConstrainedGrangerModel is None:
        raise SkipTest("scikit-learn is not installed")

    rng = np.random.default_rng(321)
    X = rng.normal(size=(40, 6)).astype(np.float64)
    y = rng.normal(size=(40, 1)).astype(np.float64)

    constraint = build_numpy_constraint_from_relations(
        relations={
            ("y1", "x1"): 0,
            ("y1", "x2"): 0.3,
        },
        predictor_names=["x1", "x2"],
        output_names=["y1"],
        col_offsets=[0, 3],
        n_features=6,
        base_mask=np.ones((1, 6), dtype=np.float64),
    )

    model = ScikitConstrainedGrangerModel(
        constraint=constraint,
        max_iter=5,
        learning_rate=0.1,
        batch_size=16,
    )
    model.initialize(X, lags=1, targets=y)
    result = model.fit()

    assert "weights" in result
    coef = model.coef_
    # first predictor block must remain hard-zero due to constraint mask
    np.testing.assert_allclose(coef[0, 0:3], 0.0, atol=1e-9)


def test_pytorch_constraint_enforces_mask_and_min_abs_sum_and_model_compatibility():
    _require_torch()

    import torch
    from ..backends.models.pytorch_model import PyTorchGrangerModel

    constraint = build_pytorch_constraint_from_relations(
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

    w = torch.zeros((1, 6), dtype=torch.float32)
    constrained = constraint(w)

    np.testing.assert_allclose(constrained[:, 0:3].detach().cpu().numpy(), 0.0)
    sum_abs = float(torch.sum(torch.abs(constrained[:, 3:6])).item())
    assert sum_abs >= 1.5 - 1e-6
    assert constraint.is_satisfied(constrained)

    # model compatibility: _apply_constraint expects callable returning tensor same shape
    rng = np.random.default_rng(456)
    X = rng.normal(size=(24, 6)).astype(np.float64)
    y = rng.normal(size=(24, 1)).astype(np.float64)

    model = PyTorchGrangerModel(
        constraint=constraint,
        epochs=1,
        batch_size=8,
        learning_rate=0.01,
        device="cpu",
        verbose=0,
    )
    model.initialize(X, lags=1, targets=y)

    with torch.no_grad():
        model._coefficient_layer.weight.zero_()  # pylint: disable=protected-access
    model._apply_constraint()  # pylint: disable=protected-access

    constrained_model_w = model._coefficient_layer.weight.detach().cpu().numpy()  # pylint: disable=protected-access
    np.testing.assert_allclose(constrained_model_w[:, 0:3], 0.0, atol=1e-8)
    assert float(np.sum(np.abs(constrained_model_w[:, 3:6]))) >= 1.5 - 1e-6


if __name__ == "__main__":
    tests = [
        test_numpy_constraint_enforces_mask_and_min_abs_sum,
        test_numpy_constraint_is_compatible_with_scikit_model,
        test_pytorch_constraint_enforces_mask_and_min_abs_sum_and_model_compatibility,
    ]

    print("\n" + "=" * 80)
    print("CONSTRAINT TESTS")
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
