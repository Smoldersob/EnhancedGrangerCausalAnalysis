from .base_model import BaseGrangerModel
import importlib

__all__ = ["BaseGrangerModel"]

# Import optional backend models defensively: if import fails for any reason
# (missing binary wheels, incompatible versions), we silently skip exposing
# that backend to avoid breaking top-level imports during test collection.
try:
    if importlib.util.find_spec("sklearn") is not None:
        from .scikit_model import ScikitConstrainedGrangerModel
        __all__.append("ScikitConstrainedGrangerModel")
        # Backwards compatibility: tests and older code expect `SklearnGrangerModel`.
        SklearnGrangerModel = ScikitConstrainedGrangerModel
        __all__.append("SklearnGrangerModel")
except Exception:
    ScikitConstrainedGrangerModel = None

try:
    if importlib.util.find_spec("tensorflow") is not None:
        from .tensorflow_model import TensorFlowGrangerModel
        __all__.append("TensorFlowGrangerModel")
except Exception:
    TensorFlowGrangerModel = None

try:
    if importlib.util.find_spec("torch") is not None:
        from .pytorch_model import PyTorchGrangerModel
        __all__.append("PyTorchGrangerModel")
except Exception:
    PyTorchGrangerModel = None