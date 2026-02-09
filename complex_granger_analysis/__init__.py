import importlib
from .granger_tests.complex_granger import ComplexGrangerAnalisysModel
from .granger_tests.scikit_granger import SparseConstaraintedMVGC, MTCLR
from .granger_tests.statsmodels_granger import grangers_causation_matrix
from .callbacks.callbacks import Callback,ProcentageChange,EarlyStopping

if importlib.util.find_spec("statsmodels") is None:
    raise ImportError("statsmodels is required")

__all__ = ['ComplexGrangerAnalisysModel','grangers_causation_matrix', 'Callback','ProcentageChange', 'EarlyStopping','auto_select_lag']

if importlib.util.find_spec("sklearn") is not None:
    __all__.extend(['SparseConstaraintedMVGC','MTCLR'])

if importlib.util.find_spec("tensorflow") is not None:
    from .granger_tests.tensorflow_granger import TFNeuralSparseConstaraintedMVGC
    from .regularizers.regularizers_keras import KerasCyclicL1Regularizer
    __all__.extend(['TFNeuralSparseConstaraintedMVGC','KerasCyclicL1Regularizer'])

if importlib.util.find_spec("torch") is not None:
    from .granger_tests.pytorch_granger import PTNeuralSparseConstaraintedMVGC
    from .regularizers.regularizers_pytorch import CyclicL1Regularizer
    __all__.extend(['PTNeuralSparseConstaraintedMVGC','CyclicL1Regularizer'])

__version__ = '1.0.0'
