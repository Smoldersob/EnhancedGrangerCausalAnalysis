from collections.abc import Sequence
from typing import Any

from .._spec import extract_typed_spec


def resolve_callback(spec: Any) -> Any:
    import keras

    callback_base = keras.callbacks.Callback

    if isinstance(spec, callback_base):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, callback_base):
            raise TypeError(
                "Callback class must inherit from keras.callbacks.Callback."
            )
        return spec()

    name, params = extract_typed_spec(spec, context="callback")

    aliases = {
        "early_stopping": "EarlyStopping",
        "keras_early_stopping": "EarlyStopping",
        "earlystopping": "EarlyStopping",
        "reduce_lr_on_plateau": "ReduceLROnPlateau",
        "reduce_learning_rate": "ReduceLROnPlateau",
        "reduce_lr": "ReduceLROnPlateau",
        "reducelronplateau": "ReduceLROnPlateau",
        "keras_reduce_lr": "ReduceLROnPlateau",
        "tensorboard": "TensorBoard",
        "keras_tensorboard": "TensorBoard",
        "model_checkpoint": "ModelCheckpoint",
        "checkpoint": "ModelCheckpoint",
        "keras_checkpoint": "ModelCheckpoint",
        "csv_logger": "CSVLogger",
        "csvlogger": "CSVLogger",
        "keras_csv_logger": "CSVLogger",
        "terminate_on_nan": "TerminateOnNaN",
        "keras_terminate_on_nan": "TerminateOnNaN",
    }

    class_name = aliases.get(name.lower())
    if class_name is None:
        raise ValueError(
            f"Unsupported Keras callback {name!r}. "
            f"Available aliases: {', '.join(sorted(aliases))}."
        )

    callback_class = getattr(keras.callbacks, class_name)
    return callback_class(**params)


def resolve_callbacks(specs: Any) -> list[Any] | None:
    if specs is None:
        return None

    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
        specs = [specs]

    callbacks = [resolve_callback(spec) for spec in specs]
    return callbacks or None