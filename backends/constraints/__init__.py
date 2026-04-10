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

