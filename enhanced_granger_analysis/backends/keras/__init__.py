import importlib
if importlib.util.find_spec("keras") is not None:
    try:
        from . import constraints, regularizers
        from .keras_backend import KerasBackendStrategy

        __all__ = [
            "constraints",
            "regularizers",
            "KerasBackendStrategy",
        ]
    except ImportError as e:
        print(f"Error importing Keras backend: {e}")
        __all__ = []
else:
    print("Keras not found, skipping Keras backend.")
    __all__ = []