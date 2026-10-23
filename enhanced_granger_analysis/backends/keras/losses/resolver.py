from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_ALIASES = {
    "mse": "MeanSquaredError",
    "mean_squared_error": "MeanSquaredError",
    "mae": "MeanAbsoluteError",
    "mean_absolute_error": "MeanAbsoluteError",
    "mape": "MeanAbsolutePercentageError",
    "mean_absolute_percentage_error": "MeanAbsolutePercentageError",
    "msle": "MeanSquaredLogarithmicError",
    "mean_squared_logarithmic_error": "MeanSquaredLogarithmicError",
    "huber": "Huber",
    "log_cosh": "LogCosh",
    "bce": "BinaryCrossentropy",
    "binary_crossentropy": "BinaryCrossentropy",
    "categorical_crossentropy": "CategoricalCrossentropy",
    "sparse_categorical_crossentropy": "SparseCategoricalCrossentropy",
}


def _normalize_name(name: str) -> str:
    return name.strip().replace("_", "").replace("-", "").lower()


def _get_loss_class(name: str) -> type:
    import keras

    if not name.strip():
        raise ValueError("Loss name must not be empty.")

    class_name = _ALIASES.get(name.strip().lower(), name.strip())
    normalized = _normalize_name(class_name)

    for exported_name in dir(keras.losses):
        if _normalize_name(exported_name) != normalized:
            continue

        candidate = getattr(keras.losses, exported_name)

        if (
            isinstance(candidate, type)
            and issubclass(candidate, keras.losses.Loss)
            and candidate is not keras.losses.Loss
        ):
            return candidate

    raise ValueError(
        f"Unknown Keras loss class {name!r}. "
        "Use a built-in loss name or pass a custom class, "
        "instance or function directly."
    )


def resolve_loss(spec: Any = None) -> Any:
    """Return a ready Keras loss or None.

    Accepted:
        None;
        built-in name;
        {"type": name, "params": {...}};
        Loss instance;
        concrete Loss class;
        loss function or callable object.

    Callable factories are not invoked.
    """
    if spec is None:
        return None

    import keras

    if isinstance(spec, keras.losses.Loss):
        return spec

    if isinstance(spec, type):
        if (
            not issubclass(spec, keras.losses.Loss)
            or spec is keras.losses.Loss
        ):
            raise TypeError(
                "Loss class must be a concrete subclass "
                "of keras.losses.Loss."
            )

        return spec()

    if isinstance(spec, str):
        return _get_loss_class(spec)()

    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "params"}
        if unknown:
            raise ValueError(
                f"Unknown loss specification fields: {unknown!r}."
            )

        name = spec.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "loss.type must be a non-empty string."
            )

        params = spec.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError("loss.params must be a mapping.")

        if any(not isinstance(key, str) for key in params):
            raise TypeError(
                "Loss parameter names must be strings."
            )

        return _get_loss_class(name)(**dict(params))

    if callable(spec):
        return spec

    raise TypeError(
        "Loss must be None, a name, a specification mapping, "
        "a Keras Loss instance, a Loss class or a function."
    )