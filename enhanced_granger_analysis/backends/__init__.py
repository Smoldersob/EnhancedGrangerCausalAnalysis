import importlib.util

from .base_backend import BackendStrategy
from .backend_factory import BackendFactory

__all__ = [
	"BackendStrategy",
	"BackendFactory",
]

if importlib.util.find_spec("tensorflow") is not None:
	try:
		from .tensorflow_backend import TensorFlowBackendStrategy
		__all__.append("TensorFlowBackendStrategy")
	except Exception:
		TensorFlowBackendStrategy = None

if importlib.util.find_spec("torch") is not None:
	try:
		from .pytorch_backend import PyTorchBackendStrategy
		__all__.append("PyTorchBackendStrategy")
	except Exception:
		PyTorchBackendStrategy = None

if importlib.util.find_spec("sklearn") is not None:
	try:
		from .scikit_backend import ScikitBackendStrategy
		__all__.append("ScikitBackendStrategy")
	except Exception:
		ScikitBackendStrategy = None

try:
	from . import callbacks, models, constraints, regularizers
	__all__.extend(["callbacks", "models", "constraints", "regularizers"])
except Exception:
	# If optional backend imports fail during package import (e.g. broken torch),
	# avoid raising so top-level imports remain importable for tests that skip
	# optional-backend functionality.
	callbacks = None
	models = None
	constraints = None
	regularizers = None