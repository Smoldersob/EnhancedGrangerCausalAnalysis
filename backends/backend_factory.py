"""Factory for instantiating backend strategies."""

import importlib.util
from typing import Dict, List, Optional

from .base_backend import BackendStrategy
from .pytorch_backend import PyTorchBackendStrategy
from .scikit_backend import ScikitBackendStrategy
from .tensorflow_backend import TensorFlowBackendStrategy


class BackendFactory:
	"""
	Factory for obtaining and managing backend strategies.

	Traits:
	- Enumerates all available backends (TensorFlow, PyTorch, scikit-learn).
	- Provides lazy instantiation of strategies.
	- Returns preferred backend if multiple are available.
	- Validates backend availability before use.
	"""

	# Registry of known strategies (name -> class) enabled by optional dependencies.
	_BACKEND_REGISTRY: Dict[str, type] = {}
	if importlib.util.find_spec("tensorflow") is not None:
		_BACKEND_REGISTRY.update(
			{
				"tensorflow": TensorFlowBackendStrategy,
				"tf": TensorFlowBackendStrategy,
				"keras": TensorFlowBackendStrategy,
			}
		)
	if importlib.util.find_spec("torch") is not None:
		_BACKEND_REGISTRY.update(
			{
				"pytorch": PyTorchBackendStrategy,
				"torch": PyTorchBackendStrategy,
			}
		)
	if importlib.util.find_spec("sklearn") is not None:
		_BACKEND_REGISTRY.update(
			{
				"sklearn": ScikitBackendStrategy,
				"scikit": ScikitBackendStrategy,
				"scikit-learn": ScikitBackendStrategy,
			}
		)

	# Cache of instantiated strategies
	_STRATEGY_CACHE: Dict[str, Optional[BackendStrategy]] = {}

	@classmethod
	def get_strategy(
		cls,
		backend_name: Optional[str] = None,
	) -> BackendStrategy:
		"""
		Retrieve or instantiate a backend strategy by name.

		Parameters
		----------
		backend_name : str or None
			Name of backend: 'tensorflow'/'tf'/'keras', 'pytorch'/'torch', 'sklearn'/'scikit-learn'.
			If None, returns the first available backend in preference order.

		Returns
		-------
		BackendStrategy
			Instantiated strategy for the requested backend.

		Raises
		------
		ValueError
			If backend_name is not recognized or no backends are available.
		"""
		if backend_name is None:
			# Return first available backend in preference order
			backend_name = cls._get_preferred_available_backend()

		backend_name_lower = backend_name.lower()
		if backend_name_lower not in cls._BACKEND_REGISTRY:
			available = cls.list_available_backends()
			raise ValueError(
				f"Unknown backend '{backend_name}'. "
				f"Available: {available if available else 'none (check dependencies)'}"
			)

		# Check cache
		if backend_name_lower in cls._STRATEGY_CACHE:
			cached = cls._STRATEGY_CACHE[backend_name_lower]
			if cached is not None:
				return cached
			raise ValueError(f"Backend '{backend_name}' is not available.")

		# Instantiate strategy
		strategy_cls = cls._BACKEND_REGISTRY[backend_name_lower]
		strategy = strategy_cls()

		if not strategy.is_available():
			cls._STRATEGY_CACHE[backend_name_lower] = None
			raise ValueError(
				f"Backend '{backend_name}' is not available. "
				f"Please install the required package (tensorflow, torch, or scikit-learn)."
			)

		cls._STRATEGY_CACHE[backend_name_lower] = strategy
		return strategy

	@classmethod
	def is_backend_available(cls, backend_name: str) -> bool:
		"""
		Check if a backend is available without raising exceptions.

		Parameters
		----------
		backend_name : str
			Name of backend to check.

		Returns
		-------
		bool
			True if backend is available, False otherwise.
		"""
		backend_name_lower = backend_name.lower()
		if backend_name_lower not in cls._BACKEND_REGISTRY:
			return False

		# Check cache first
		if backend_name_lower in cls._STRATEGY_CACHE:
			return cls._STRATEGY_CACHE[backend_name_lower] is not None

		# Try instantiation
		try:
			strategy_cls = cls._BACKEND_REGISTRY[backend_name_lower]
			strategy = strategy_cls()
			available = strategy.is_available()
			cls._STRATEGY_CACHE[backend_name_lower] = strategy if available else None
			return available
		except Exception:
			cls._STRATEGY_CACHE[backend_name_lower] = None
			return False

	@classmethod
	def list_available_backends(cls) -> List[str]:
		"""
		List all currently available backends.

		Returns
		-------
		list of str
			Backend names that are installed and usable.
		"""
		available = []
		for backend_name in list(set(cls._BACKEND_REGISTRY.values())):
			# Get canonical name for each strategy class
			canonical_name = None
			for alias, strategy_cls in cls._BACKEND_REGISTRY.items():
				if strategy_cls.__name__ == backend_name.__name__:
					canonical_name = alias
					break

			if canonical_name and cls.is_backend_available(canonical_name):
				# Avoid duplicates (TensorFlow registered under multiple aliases)
				if canonical_name not in available:
					available.append(canonical_name)

		return available

	@classmethod
	def _get_preferred_available_backend(cls) -> str:
		"""
		Return the name of the first available backend in preference order.

		Preference order: PyTorch > TensorFlow > scikit-learn

		Returns
		-------
		str
			Name of the preferred available backend.

		Raises
		------
		ValueError
			If no backends are available.
		"""
		preference_order = ["pytorch", "tensorflow", "sklearn"]

		for backend_name_pref in preference_order:
			if cls.is_backend_available(backend_name_pref):
				return backend_name_pref

		raise ValueError(
			"No machine learning backends are available. "
			"Install at least one optional dependency: tensorflow, torch, scikit-learn."
		)

	@classmethod
	def reset_cache(cls) -> None:
		"""Clear the strategy cache (useful for testing)."""
		cls._STRATEGY_CACHE.clear()
