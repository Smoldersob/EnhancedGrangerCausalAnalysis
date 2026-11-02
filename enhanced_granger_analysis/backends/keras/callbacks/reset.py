from __future__ import annotations

import copy
import inspect
import os
import warnings
from collections.abc import Mapping
from typing import Any

import keras

from ....core.exceptions import ConstraintConfigurationError
from ....core.components.callback_lifecycle import (
    clone_custom_callback,
    reset_callback_list,
    validate_run_name,
)


_TENSORBOARD_ROOT_ATTR = "_egca_tensorboard_log_root"


def _apply_config_updates(
    config: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = set(updates) - set(config)
    if unknown:
        raise ConstraintConfigurationError(
            "Unsupported callback configuration keys: "
            f"{sorted(unknown)}"
        )

    result = copy.deepcopy(dict(config))
    result.update(copy.deepcopy(dict(updates)))
    return result


def _read_tensorboard_profile_batch(
    callback: keras.callbacks.TensorBoard,
) -> Any:
    """Recover profiling configuration, not profiling runtime state."""
    if hasattr(callback, "profile_batch"):
        return copy.deepcopy(callback.profile_batch)

    # Compatibility adapter for implementations storing the configured
    # profiling interval as private fields.
    if (
        hasattr(callback, "_start_batch")
        and hasattr(callback, "_stop_batch")
    ):
        start = int(callback._start_batch)
        stop = int(callback._stop_batch)

        if start == 0 and stop == 0:
            return 0
        if start == stop:
            return start
        return (start, stop)

    raise ConstraintConfigurationError(
        "Cannot recover TensorBoard profile_batch configuration "
        "for this Keras version. Supply profile_batch explicitly "
        "in callback_updates or use RunTensorBoard."
    )


def _tensorboard_constructor_parameters() -> set[str]:
    parameters = inspect.signature(
        keras.callbacks.TensorBoard.__init__
    ).parameters

    return {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def _clone_tensorboard(
    callback: keras.callbacks.TensorBoard,
    run_name: str,
    updates: Mapping[str, Any],
) -> keras.callbacks.TensorBoard:
    constructor_parameters = _tensorboard_constructor_parameters()

    # log_root is an adapter alias, not a TensorBoard constructor argument.
    allowed_updates = constructor_parameters | {"log_root"}
    unknown = set(updates) - allowed_updates
    if unknown:
        raise ConstraintConfigurationError(
            "Unsupported TensorBoard configuration keys: "
            f"{sorted(unknown)}"
        )
    if "log_root" in updates and "log_dir" in updates:
        raise ConstraintConfigurationError(
            "Use log_root or log_dir, not both."
        )

    config: dict[str, Any] = {}

    for name in constructor_parameters:
        if name == "log_dir":
            continue

        if name in updates:
            config[name] = copy.deepcopy(updates[name])
        elif name == "profile_batch":
            config[name] = _read_tensorboard_profile_batch(callback)
        elif hasattr(callback, name):
            config[name] = copy.deepcopy(getattr(callback, name))
        else:
            raise ConstraintConfigurationError(
                f"Cannot recover TensorBoard parameter {name!r} "
                "for this Keras version. Supply it explicitly "
                "in callback_updates or use RunTensorBoard."
            )

    # The first ordinary TensorBoard.log_dir becomes the stable root.
    # Clones carry that root into subsequent resets.
    root = getattr(
        callback,
        _TENSORBOARD_ROOT_ATTR,
        callback.log_dir,
    )

    if "log_root" in updates:
        root = updates["log_root"]
    elif "log_dir" in updates:
        root = updates["log_dir"]

    root = os.fspath(root)
    if not root:
        raise ConstraintConfigurationError(
            "TensorBoard log root must not be empty."
        )

    # Use forward slashes for URI-like log roots as well.
    config["log_dir"] = (
        root.rstrip("/") + "/" + run_name
        if "://" in root
        else os.path.join(root, run_name)
    )

    fresh = keras.callbacks.TensorBoard(**config)
    setattr(fresh, _TENSORBOARD_ROOT_ATTR, root)
    return fresh


def _clone_lambda_callback(
    callback: keras.callbacks.LambdaCallback,
    updates: Mapping[str, Any],
) -> keras.callbacks.LambdaCallback:
    hook_names = {
        name
        for name in dir(keras.callbacks.Callback)
        if name.startswith("on_")
        and callable(getattr(keras.callbacks.Callback, name))
    }

    unknown = set(updates) - hook_names
    if unknown:
        raise ConstraintConfigurationError(
            "LambdaCallback updates must refer to callback hooks: "
            f"{sorted(unknown)}"
        )

    # LambdaCallback installs its supplied hooks as instance attributes.
    hooks = {
        name: function
        for name, function in vars(callback).items()
        if name in hook_names and callable(function)
    }

    for name, function in updates.items():
        if not callable(function):
            raise ConstraintConfigurationError(
                f"LambdaCallback hook {name!r} must be callable."
            )
        hooks[name] = function

    warnings.warn(
        "LambdaCallback is reconstructed, but its functions are reused. "
        "External state held by closures, callable objects or resources "
        "is not reset. Use a custom clone_for_run() when needed.",
        UserWarning,
        stacklevel=3,
    )

    return keras.callbacks.LambdaCallback(**hooks)


def clone_keras_callback(
    callback: keras.callbacks.Callback,
    run_name: str,
    updates: Mapping[str, Any],
) -> keras.callbacks.Callback:
    validate_run_name(run_name)

    if not isinstance(callback, keras.callbacks.Callback):
        raise ConstraintConfigurationError(
            "Keras callbacks must inherit from keras.callbacks.Callback."
        )

    if callable(getattr(callback, "clone_for_run", None)):
        return clone_custom_callback(callback, run_name, updates)

    # Exact types only. Custom subclasses need clone_for_run().
    callback_type = type(callback)

    if callback_type is keras.callbacks.TensorBoard:
        return _clone_tensorboard(callback, run_name, updates)

    if callback_type is keras.callbacks.EarlyStopping:
        config = {
            "monitor": callback.monitor,
            "min_delta": abs(callback.min_delta),
            "patience": callback.patience,
            "verbose": callback.verbose,
            "mode": callback.mode,
            "baseline": callback.baseline,
            "restore_best_weights": callback.restore_best_weights,
            "start_from_epoch": callback.start_from_epoch,
        }
        return keras.callbacks.EarlyStopping(
            **_apply_config_updates(config, updates)
        )

    if callback_type is keras.callbacks.ReduceLROnPlateau:
        config = {
            "monitor": callback.monitor,
            "factor": callback.factor,
            "patience": callback.patience,
            "verbose": callback.verbose,
            "mode": callback.mode,
            "min_delta": abs(callback.min_delta),
            "cooldown": callback.cooldown,
            "min_lr": callback.min_lr,
        }
        return keras.callbacks.ReduceLROnPlateau(
            **_apply_config_updates(config, updates)
        )

    if callback_type is keras.callbacks.LearningRateScheduler:
        config = {
            "schedule": callback.schedule,
            "verbose": callback.verbose,
        }
        if set(updates) - set(config):
            raise ConstraintConfigurationError(
                "LearningRateScheduler supports only schedule "
                "and verbose updates."
            )

        # Preserve the callable instead of attempting to deepcopy it.
        config.update(dict(updates))
        if not callable(config["schedule"]):
            raise ConstraintConfigurationError(
                "LearningRateScheduler.schedule must be callable."
            )

        return keras.callbacks.LearningRateScheduler(**config)

    if callback_type is keras.callbacks.RemoteMonitor:
        config = {
            "root": callback.root,
            "path": callback.path,
            "field": callback.field,
            "headers": callback.headers,
            "send_as_json": callback.send_as_json,
        }
        return keras.callbacks.RemoteMonitor(
            **_apply_config_updates(config, updates)
        )

    if callback_type is keras.callbacks.LambdaCallback:
        return _clone_lambda_callback(callback, updates)

    if callback_type is keras.callbacks.TerminateOnNaN:
        if updates:
            raise ConstraintConfigurationError(
                "TerminateOnNaN has no supported configuration updates."
            )
        return keras.callbacks.TerminateOnNaN()

    if callback_type is keras.callbacks.Callback:
        if updates:
            raise ConstraintConfigurationError(
                "Base Callback has no supported configuration updates."
            )
        return keras.callbacks.Callback()

    raise ConstraintConfigurationError(
        f"{callback.__class__.__name__} has no supported reset recipe. "
        "Implement clone_for_run(run_name, callback_updates=None) "
        "in a keras.callbacks.Callback subclass."
    )

def reset_callbacks_for_run(
    callbacks: list[keras.callbacks.Callback],
    *,
    run_name: str,
    callback_updates: Mapping | None = None,
) -> list[keras.callbacks.Callback]:
    """Return fresh callbacks without mutating the source list."""
    return reset_callback_list(
        callbacks,
        run_name,
        callback_updates,
        expected_type=keras.callbacks.Callback,
        clone_callback=clone_keras_callback,
    )


