import importlib
from .complex_granger import ComplexGrangerAnalisysModel
from .statsmodels_granger import grangers_causation_matrix

__all__ = ['ComplexGrangerAnalisysModel','grangers_causation_matrix']

if importlib.util.find_spec("sklearn") is not None:
    from .scikit_granger import SparseConstaraintedMVGC
    __all__.extend(['SparseConstaraintedMVGC'])

if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_granger import TFNeuralSparseConstaraintedMVGC
    __all__.extend(['TFNeuralSparseConstaraintedMVGC'])

if importlib.util.find_spec("torch") is not None:
    from .pytorch_granger import PTNeuralSparseConstaraintedMVGC
    __all__.extend(['PTNeuralSparseConstaraintedMVGC'])