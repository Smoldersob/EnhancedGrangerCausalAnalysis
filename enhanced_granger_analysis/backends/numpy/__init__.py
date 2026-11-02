import importlib
if importlib.util.find_spec("numpy") is not None:
    try:
        from . import constraints, regularizers
        from .numpy_backend import NumpyBackendStrategy

        __all__ = [
            "constraints",
            "regularizers",
            "NumpyBackendStrategy",
        ]
    except ImportError as e:
        print(f"Error importing Numpy backend: {e}")
        __all__ = []
else:
    print("Numpy not found, skipping Numpy backend.")
    __all__ = []