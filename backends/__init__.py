import importlib.util

from .base_backend import BackendStrategy
from .backend_factory import BackendFactory

__all__ = [
	"BackendStrategy",
	"BackendFactory",
]

if importlib.util.find_spec("tensorflow") is not None:
	from .tensorflow_backend import TensorFlowBackendStrategy

	__all__.append("TensorFlowBackendStrategy")

if importlib.util.find_spec("torch") is not None:
	from .pytorch_backend import PyTorchBackendStrategy

	__all__.append("PyTorchBackendStrategy")

if importlib.util.find_spec("sklearn") is not None:
	from .scikit_backend import ScikitBackendStrategy

	__all__.append("ScikitBackendStrategy")

from . import callbacks, models, constraints, regularizers
__all__.extend(["callbacks", "models", "constraints", "regularizers"])