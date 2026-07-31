import importlib.util

__all__ = []

if importlib.util.find_spec("pandas") is not None:
	try:
		from ..hiperopt.hyperoptimization import HyperoptimizationResult, MultiTaskGrangerHyperparameterOptimizer
		
		__all__.extend([
			"HyperoptimizationResult",
			"MultiTaskGrangerHyperparameterOptimizer",
		])
	except Exception:
		# Keep lightweight utilities importable even if optional backend stack
		# cannot be imported in a given environment.
		pass