import importlib
__all__ = []
if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_regularizers import KerasL1Regularizer, KerasLagDependentL1Regularizer
    __all__.extend(["KerasL1Regularizer", "KerasLagDependentL1Regularizer"])
if importlib.util.find_spec("numpy") is not None:    
    from .numpy_regularizers import NumpyL1Regularizer, NumpyLagDependentL1Regularizer
    __all__.extend(["NumpyL1Regularizer", "NumpyLagDependentL1Regularizer"])
if importlib.util.find_spec("torch") is not None:
    from .pytorch_regularizers import PyTorchL1Regularizer, PyTorchLagDependentL1Regularizer
    __all__.extend(["PyTorchL1Regularizer", "PyTorchLagDependentL1Regularizer"])

