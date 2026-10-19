import importlib.util

__all__ = []



if importlib.util.find_spec("pandas") is not None:
	try:
		from .config_loader import BuilderConfigLoader, TestGroupConfigIterator
		__all__.extend([
			"BuilderConfigLoader",
			"TestGroupConfigIterator",
		])
	except ImportError as e:
		print(f"Warning: Config loader dependencies could not be imported. "
			  f"Error details: {e}")
		
	try:
		from .builder import MultiTaskGrangerBuilder, MultitaskGrangerBuilder
		from .orchestrator import MultiTaskGrangerAPI
		from .simple_granger import SimpleGrangerAPI, SimpleGrangerOutput

		__all__.extend([
			"MultiTaskGrangerBuilder",
			"MultitaskGrangerBuilder",
			"MultiTaskGrangerAPI",
			"SimpleGrangerAPI",
			"SimpleGrangerOutput",
		])
	except Exception as e:
		# Keep lightweight utilities importable even if optional backend stack
		# cannot be imported in a given environment.
		print(f"Warning: Optional backend dependencies could not be imported. "
			  f"MultiTaskGrangerBuilder and related classes will not be available."
			  f"Error details: {e}")
		pass
