import importlib.util

from .base_backend import BackendStrategy
from .backend_factory import BackendFactory

__all__ = [
	"BackendStrategy",
	"BackendFactory",
]

if importlib.util.find_spec("keras") is not None:
	try:
		from .keras import KerasBackendStrategy
		__all__.append("KerasBackendStrategy")
		from . import keras
		__all__.append("keras")
	except Exception:
		KerasBackendStrategy = None

if importlib.util.find_spec("torch") is not None:
	try:
		from .pytorch import PyTorchBackendStrategy
		__all__.append("PyTorchBackendStrategy")
		from . import pytorch
		__all__.append("pytorch")
	except Exception:
		PyTorchBackendStrategy = None

if importlib.util.find_spec("numpy") is not None:
	try:
		from .numpy.numpy_backend import NumpyBackendStrategy
		__all__.append("NumpyBackendStrategy")
		from . import numpy 
		__all__.append("numpy")
	except Exception:
		NumpyBackendStrategy = None