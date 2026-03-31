import importlib.util

__all__ = []

from .config_loader import BuilderConfigLoader, TestGroupConfigIterator

__all__.extend([
	"BuilderConfigLoader",
	"TestGroupConfigIterator",
])

if importlib.util.find_spec("pandas") is not None:
	try:
		from .builder import GrangerAnalysisBuilder, MultitaskGrangerBuilder
		from .orchestrator import MultiTaskGrangerAPI, MultitaskGrangerOutput
		from .simple_granger import SimpleGrangerAPI, SimpleGrangerOutput

		__all__.extend([
			"MultitaskGrangerBuilder",
			"GrangerAnalysisBuilder",
			"MultiTaskGrangerAPI",
			"MultitaskGrangerOutput",
			"SimpleGrangerAPI",
			"SimpleGrangerOutput",
		])
	except Exception:
		# Keep lightweight utilities importable even if optional backend stack
		# cannot be imported in a given environment.
		pass
