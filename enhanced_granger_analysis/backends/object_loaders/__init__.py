import importlib

# Import optional backend models defensively: if import fails for any reason
# (missing binary wheels, incompatible versions), we silently skip exposing
# that backend to avoid breaking top-level imports during test collection.
__all__ = []
try:
    if importlib.util.find_spec("sklearn") is not None:
        from .np_object_loader import NumpyObjectLoader
        __all__.append("NumpyObjectLoader")
except Exception:
    pass
try:
    if importlib.util.find_spec("Keras") is not None:
        from .keras_object_loader import KerasObjectLoader
        __all__.append("KerasObjectLoader")
except Exception:
    pass

try:
    if importlib.util.find_spec("torch") is not None:
        from .torch_object_loader import TorchObjectLoader
        __all__.append("PyTorchGrangerModel")
except Exception:
    pass