import importlib
from .complex_granger import ComplexGrangerAnalysisModel
from .statsmodels_granger import grangers_causation_matrix

__all__ = ['ComplexGrangerAnalysisModel','grangers_causation_matrix']

if importlib.util.find_spec("sklearn") is not None:
    from .scikit_granger import SparseConstrainedMVGC
    __all__.extend(['SparseConstrainedMVGC'])

if importlib.util.find_spec("tensorflow") is not None:
    from .tensorflow_granger import TFNeuralSparseConstrainedMVGC
    __all__.extend(['TFNeuralSparseConstrainedMVGC'])

if importlib.util.find_spec("torch") is not None:
    from .pytorch_granger import PTNeuralSparseConstrainedMVGC
    __all__.extend(['PTNeuralSparseConstrainedMVGC'])