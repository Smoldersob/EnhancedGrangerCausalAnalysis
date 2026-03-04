import importlib
__all__=[]

if importlib.util.find_spec("tensorflow") is not None:
    from .regularizers_keras import KerasCyclicL1Regularizer
    __all__.extend(['KerasCyclicL1Regularizer'])

if importlib.util.find_spec("torch") is not None:
    from .regularizers_pytorch import CyclicL1Regularizer
    __all__.extend(['CyclicL1Regularizer'])