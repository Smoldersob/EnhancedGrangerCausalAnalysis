import importlib.util

from . import (
	constraints_config, 
	exceptions, 
	lag_config, 
	protocols,
	training_config
)

__all__ = [
	"constraints_config",
    "exceptions",
    "lag_config",
    "protocols",
    "training_config",
]


def __getattr__(name: str):
	"""Lazy-load optional submodules to avoid circular imports at package init time."""
	if name == "outputs":
		if importlib.util.find_spec("pandas") is None:
			raise AttributeError("core.outputs requires pandas")
		from . import outputs as _outputs
		return _outputs
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
