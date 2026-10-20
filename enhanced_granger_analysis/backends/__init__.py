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

if importlib.util.find_spec("sklearn") is not None:
	try:
		from .scikit.scikit_backend import ScikitBackendStrategy
		__all__.append("ScikitBackendStrategy")
		from . import scikit 
		__all__.append("numpy")
	except Exception:
		ScikitBackendStrategy = None