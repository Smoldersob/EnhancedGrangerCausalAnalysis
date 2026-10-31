"""Analysis configuration independent of execution state."""

from .analysis_spec import AnalysisSpec
from .modeling_spec import ModelingSpec

__all__ = [
	"AnalysisSpec",
	"ModelingSpec",
]