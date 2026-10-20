import importlib
if importlib.util.find_spec("torch") is not None:
    try:
        from . import constraints, regularizers
        from .pytorch_backend import PyTorchBackendStrategy

        __all__ = [
            "constraints",
            "regularizers",
            "PyTorchBackendStrategy",
        ]
    except ImportError as e:
        print(f"Error importing PyTorch backend: {e}")
        __all__ = []
else:
    print("PyTorch not found, skipping PyTorch backend.")
    __all__ = []