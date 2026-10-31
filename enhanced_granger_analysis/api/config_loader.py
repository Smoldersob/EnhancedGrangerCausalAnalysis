"""Compatibility imports for configuration loading and case expansion."""

from ..configuration.loader import AnalysisConfigLoader
from ..experiments.config_iterator import TestGroupConfigIterator

BuilderConfigLoader = AnalysisConfigLoader

__all__ = [
    "AnalysisConfigLoader",
    "BuilderConfigLoader",
    "TestGroupConfigIterator",
]