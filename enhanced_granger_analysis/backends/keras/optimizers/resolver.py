from collections.abc import Mapping
from typing import Any

from .._spec import extract_typed_spec


def resolve_optimizer(spec: Any) -> Any:
    import keras

    optimizer_base = keras.optimizers.Optimizer

    if spec is None:
        return keras.optimizers.Adam()

    if isinstance(spec, optimizer_base):
        return spec

    if isinstance(spec, str):
        return keras.optimizers.get(spec)

    if isinstance(spec, Mapping):
        if "class_name" in spec:
            return keras.optimizers.get(dict(spec))

        name, params = extract_typed_spec(spec, context="optimizer")

        # Pobranie kanonicznej nazwy wbudowanego optymalizatora.
        prototype = keras.optimizers.get(name)
        native_config = keras.optimizers.serialize(prototype)
        native_config["config"].update(params)

        return keras.optimizers.deserialize(native_config)

    if isinstance(spec, type):
        if not issubclass(spec, optimizer_base):
            raise TypeError(
                "Optimizer class must inherit from "
                "keras.optimizers.Optimizer."
            )
        return spec()

    if callable(spec):
        optimizer = spec()
        if not isinstance(optimizer, optimizer_base):
            raise TypeError(
                "Optimizer factory must return "
                "keras.optimizers.Optimizer."
            )
        return optimizer

    raise TypeError(
        "Optimizer must be None, a name, configuration mapping, "
        "optimizer instance, class, or factory."
    )