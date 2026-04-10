from __future__ import annotations

import importlib
from typing import Any

from .__version__ import __version__

_LAZY_SUBMODULES = {
    "api",
    "backends",
    "core",
    "preprocessing",
    "utilities",
    "results",
    "initializers",
}

_LAZY_SYMBOLS = {
    "BackendFactory": (".backends", "BackendFactory"),
    "BackendStrategy": (".backends", "BackendStrategy"),
    "BuilderConfigLoader": (".api", "BuilderConfigLoader"),
    "TestGroupConfigIterator": (".api", "TestGroupConfigIterator"),
    "MultitaskGrangerBuilder": (".api", "MultitaskGrangerBuilder"),
    "MultiTaskGrangerAPI": (".api", "MultiTaskGrangerAPI"),
    "SimpleGrangerAPI": (".api", "SimpleGrangerAPI"),
}

__all__ = [
    "__version__",
    "api",
    "preprocessing",
    "backends",
    "core",
    "utilities",
    "results",
    "initializers",
    "callbacks",
    "BackendFactory",
    "BackendStrategy",
    "BuilderConfigLoader",
    "TestGroupConfigIterator",
    "GrangerAnalysisBuilder",
    "MultiTaskGrangerAPI",
    "SimpleGrangerAPI",
]


def __getattr__(name: str) -> Any:
    if name == "callbacks":
        # Backward-compatible alias to shared backend callback implementations.
        return importlib.import_module(".backends.callbacks", __name__)

    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

    if name in _LAZY_SYMBOLS:
        module_name, symbol_name = _LAZY_SYMBOLS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, symbol_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))