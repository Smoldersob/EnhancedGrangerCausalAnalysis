"""Factory for instantiating backend strategies."""

import importlib.util
from typing import Dict, List, Optional, Set

from .base_backend import BackendStrategy
from ..core.exceptions import BackendNotAvailableError, BackendSelectionError

try:
	from .tensorflow_backend import TensorFlowBackendStrategy
except Exception:  # pragma: no cover - optional backend module
	TensorFlowBackendStrategy = None  # type: ignore[assignment]

try:
	from .pytorch_backend import PyTorchBackendStrategy
except Exception:  # pragma: no cover - optional backend module
	PyTorchBackendStrategy = None  # type: ignore[assignment]

try:
	from .scikit_backend import ScikitBackendStrategy
except Exception:  # pragma: no cover - optional backend module
	ScikitBackendStrategy = None  # type: ignore[assignment]


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
	_CANONICAL_ALIASES: Dict[str, str] = {
		"tensorflow": "tensorflow",
		"tf": "tensorflow",
		"keras": "tensorflow",
		"pytorch": "pytorch",
		"torch": "pytorch",
		"sklearn": "sklearn",
		"scikit": "sklearn",
		"scikit-learn": "sklearn",
	}
	if importlib.util.find_spec("tensorflow") is not None and TensorFlowBackendStrategy is not None:
		_BACKEND_REGISTRY.update(
			{
				"tensorflow": TensorFlowBackendStrategy,
				"tf": TensorFlowBackendStrategy,
				"keras": TensorFlowBackendStrategy,
			}
		)
	if importlib.util.find_spec("torch") is not None and PyTorchBackendStrategy is not None:
		_BACKEND_REGISTRY.update(
			{
				"pytorch": PyTorchBackendStrategy,
				"torch": PyTorchBackendStrategy,
			}
		)
	if importlib.util.find_spec("sklearn") is not None and ScikitBackendStrategy is not None:
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
		loading_verbose: bool = False,
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
		BackendSelectionError
			If backend_name is not recognized.
		BackendNotAvailableError
			If backend exists but is unavailable or no backends are available.
		"""
		if backend_name is None:
			# Return first available backend in preference order
			backend_name = cls._get_preferred_available_backend()

		backend_name_lower = backend_name.lower()
		canonical_name = cls._CANONICAL_ALIASES.get(backend_name_lower, backend_name_lower)
		if canonical_name not in cls._BACKEND_REGISTRY:
			available = cls.list_available_backends()
			raise BackendSelectionError(
				f"Unknown backend '{backend_name}'. "
				f"Available: {available if available else 'none (check dependencies)'}"
			)

		# Check cache
		if canonical_name in cls._STRATEGY_CACHE:
			cached = cls._STRATEGY_CACHE[canonical_name]
			if cached is not None:
				if hasattr(cached, "set_loading_verbose"):
					cached.set_loading_verbose(loading_verbose)
				return cached
			raise BackendNotAvailableError(f"Backend '{backend_name}' is not available.")

		# Instantiate strategy
		strategy_cls = cls._BACKEND_REGISTRY[canonical_name]
		strategy = strategy_cls(loading_verbose=loading_verbose)

		if not strategy.is_available():
			cls._STRATEGY_CACHE[canonical_name] = None
			raise BackendNotAvailableError(
				f"Backend '{backend_name}' is not available. "
				f"Please install the required package (tensorflow, torch, or scikit-learn)."
			)

		cls._STRATEGY_CACHE[canonical_name] = strategy
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
		canonical_name = cls._CANONICAL_ALIASES.get(backend_name_lower, backend_name_lower)
		if canonical_name not in cls._BACKEND_REGISTRY:
			return False

		# Check cache first
		if canonical_name in cls._STRATEGY_CACHE:
			return cls._STRATEGY_CACHE[canonical_name] is not None

		# Try instantiation
		try:
			strategy_cls = cls._BACKEND_REGISTRY[canonical_name]
			strategy = strategy_cls()
			available = strategy.is_available()
			cls._STRATEGY_CACHE[canonical_name] = strategy if available else None
			return available
		except Exception:
			cls._STRATEGY_CACHE[canonical_name] = None
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
		preferred_order = ["pytorch", "tensorflow", "sklearn"]
		available: List[str] = []
		seen: Set[str] = set()
		for backend_name in preferred_order:
			if backend_name in seen:
				continue
			if cls.is_backend_available(backend_name):
				available.append(backend_name)
				seen.add(backend_name)

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
		BackendNotAvailableError
			If no backends are available.
		"""
		preference_order = ["pytorch", "tensorflow", "sklearn"]

		for backend_name_pref in preference_order:
			if cls.is_backend_available(backend_name_pref):
				return backend_name_pref

		raise BackendNotAvailableError(
			"No machine learning backends are available. "
			"Install at least one optional dependency: tensorflow, torch, scikit-learn."
		)

	@classmethod
	def reset_cache(cls) -> None:
		"""Clear the strategy cache (useful for testing)."""
		cls._STRATEGY_CACHE.clear()
