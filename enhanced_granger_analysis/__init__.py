"""Public interface for Enhanced Granger Causal Analysis.

Public symbols and subpackages are imported on demand. Configuration,
model construction, execution, and result evaluation remain separate
parts of the package.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import Any


try:
    __version__ = distribution_version("enhanced_granger_analysis")
except PackageNotFoundError:
    __version__ = "0+unknown"


_LAZY_SUBMODULES = frozenset({
    "api",
    "backends",
    "configuration",
    "core",
    "execution",
    "experiments",
    "hiperopt",
    "initializers",
    "modeling",
    "preprocessing",
    "results",
    "utilities",
})


_LAZY_SYMBOLS: dict[str, tuple[str, str]] = {
    # User-facing analysis interfaces.
    "MultiTaskGrangerBuilder": (
        ".api.builder",
        "MultiTaskGrangerBuilder",
    ),
    "MultitaskGrangerBuilder": (
        ".api.builder",
        "MultitaskGrangerBuilder",
    ),
    "MultiTaskGrangerAPI": (
        ".api.orchestrator",
        "MultiTaskGrangerAPI",
    ),
    "SimpleGrangerAPI": (
        ".api.simple_granger",
        "SimpleGrangerAPI",
    ),

    # Configuration.
    "AnalysisSpec": (
        ".configuration.analysis_spec",
        "AnalysisSpec",
    ),
    "ModelingSpec": (
        ".configuration.modeling_spec",
        "ModelingSpec",
    ),
    "AnalysisConfigLoader": (
        ".configuration.loader",
        "AnalysisConfigLoader",
    ),

    # Compatibility name for the document loader.
    "BuilderConfigLoader": (
        ".configuration.loader",
        "AnalysisConfigLoader",
    ),

    # Execution and experiment coordination.
    "AnalysisRunner": (
        ".execution.runner",
        "AnalysisRunner",
    ),
    "TestGroupConfigIterator": (
        ".experiments.config_iterator",
        "TestGroupConfigIterator",
    ),

    # Model construction and fitting services.
    "ModelAssembler": (
        ".modeling.assembler",
        "ModelAssembler",
    ),
    "ModelInitializer": (
        ".modeling.initialization",
        "ModelInitializer",
    ),
    "FitExecutor": (
        ".modeling.executor",
        "FitExecutor",
    ),
    "FitOutcome": (
        ".modeling.artifacts",
        "FitOutcome",
    ),

    # Backend interfaces.
    "BackendFactory": (
        ".backends.backend_factory",
        "BackendFactory",
    ),
    "BackendStrategy": (
        ".backends.base_backend",
        "BackendStrategy",
    ),

    # Analysis results.
    "GrangerAnalysisResults": (
        ".results.granger_results",
        "GrangerAnalysisResults",
    ),
}


__all__ = [
    "__version__",
    *sorted(_LAZY_SUBMODULES),
    *_LAZY_SYMBOLS,
]


def __getattr__(name: str) -> Any:
    """Import and cache a public attribute on first access."""

    if name in _LAZY_SUBMODULES:
        value = import_module(
            f".{name}",
            __name__,
        )
    elif name in _LAZY_SYMBOLS:
        module_name, symbol_name = _LAZY_SYMBOLS[name]

        module = import_module(
            module_name,
            __name__,
        )

        value = getattr(module, symbol_name)
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    globals()[name] = value

    return value


def __dir__() -> list[str]:
    """Expose lazy public attributes to introspection tools."""

    return sorted(
        set(globals()) | set(__all__)
    )