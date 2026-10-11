from typing import Any, Callable, Optional

_SCALERS: dict[str, Any] = {}


def register_scaler(*names: str):
    """Register aliases used by DataPreparer to resolve scaler objects."""
    def decorator(func: Callable) -> Callable:
        for name in names:
            normalized = name.strip().lower()
            if normalized in _SCALERS:
                raise ValueError(f"Scaler '{normalized}' is already registered.")
            _SCALERS[normalized] = func
        return func

    return decorator


def resolve_scaler(name: Optional[Any]) -> Any:
    """Resolve a scaler alias or accept a scaler instance/object."""
    if name is None:
        return _SCALERS.get("none")

    if isinstance(name, str):
        normalized = name.strip().lower()
        try:
            return _SCALERS[normalized]()
        except KeyError as exc:
            available = ", ".join(sorted(_SCALERS))
            raise ValueError(f"Unsupported scaler: {name}. Available scalers: {available}") from exc

    if isinstance(name, Callable):
        obj = name()
        if all(hasattr(obj, attr) and callable(getattr(obj, attr)) for attr in ("fit_transform", "transform", "inverse_transform")):
            return obj
        else:
            raise ValueError(f"Callable {name} does not return a valid scaler object.")   
    if isinstance(name, object) and all(hasattr(name, attr) and callable(getattr(name, attr)) for attr in ("fit_transform", "transform", "inverse_transform")):
        return name
    raise ValueError(f"Unsupported scaler: {name}")
    