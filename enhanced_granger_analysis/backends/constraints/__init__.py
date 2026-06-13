from .base_constaint import process_user_relations
from .numpy_constraints import (
    NumpyMaskConstraint,
    NumpyMaskAndMinAbsSumConstraint,
    build_numpy_constraint_from_relations,
)
import importlib

__all__ = [
    "process_user_relations",
    "NumpyMaskConstraint",
    "NumpyMaskAndMinAbsSumConstraint",
    "build_numpy_constraint_from_relations",
]

try:
    if importlib.util.find_spec("tensorflow") is not None:
        from .tensorflow_constraints import (
            TensorFlowMaskConstraint,
            TensorFlowMaskAndMinAbsSumConstraint,
            build_tensorflow_constraint_from_relations,
        )
        __all__.extend([
            "TensorFlowMaskConstraint",
            "TensorFlowMaskAndMinAbsSumConstraint",
            "build_tensorflow_constraint_from_relations",
        ])
except Exception:
    TensorFlowMaskConstraint = None
    TensorFlowMaskAndMinAbsSumConstraint = None
    build_tensorflow_constraint_from_relations = None

try:
    if importlib.util.find_spec("torch") is not None:
        from .pytorch_constraints import (
            PyTorchMaskConstraint,
            PyTorchMaskAndMinAbsSumConstraint,
            build_pytorch_constraint_from_relations,
        )
        __all__.extend([
            "PyTorchMaskConstraint",
            "PyTorchMaskAndMinAbsSumConstraint",
            "build_pytorch_constraint_from_relations",
        ])
except Exception:
    PyTorchMaskConstraint = None
    PyTorchMaskAndMinAbsSumConstraint = None
    build_pytorch_constraint_from_relations = None

# Provide graceful stubs when optional backends are not installed so tests
# can import symbols without requiring heavy optional dependencies at import time.
if importlib.util.find_spec("tensorflow") is None:
    def build_tensorflow_constraint_from_relations(*args, **kwargs):
        raise ImportError(
            "TensorFlow is not installed. Install 'tensorflow' to use TensorFlow constraints."
        )
    __all__.append("build_tensorflow_constraint_from_relations")

if importlib.util.find_spec("torch") is None:
    def build_pytorch_constraint_from_relations(*args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Install 'torch' to use PyTorch constraints."
        )
    if "build_pytorch_constraint_from_relations" not in __all__:
        __all__.append("build_pytorch_constraint_from_relations")

