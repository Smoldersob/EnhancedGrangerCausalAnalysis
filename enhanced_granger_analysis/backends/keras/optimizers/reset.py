from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import keras
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class OptimizerSnapshot:
    """Initial optimizer variables for one concrete optimizer instance."""

    optimizer: Any
    variables: tuple[Any, ...]
    values: tuple[NDArray[Any], ...]


def clone_optimizer(
    optimizer: keras.optimizers.Optimizer,
) -> keras.optimizers.Optimizer:
    """Create an independent optimizer from configuration, without state."""
    if not isinstance(optimizer, keras.optimizers.Optimizer):
        raise TypeError(
            "optimizer must be a keras.optimizers.Optimizer instance."
        )

    cloned = type(optimizer).from_config(optimizer.get_config())

    if not isinstance(cloned, keras.optimizers.Optimizer):
        raise TypeError(
            "Optimizer.from_config() must return a Keras optimizer."
        )

    if cloned is optimizer:
        raise ValueError(
            "Optimizer.from_config() must return a new instance."
        )

    return cloned


def capture_optimizer_state(
    optimizer: keras.optimizers.Optimizer,
) -> OptimizerSnapshot:
    """Capture variables after optimizer.build(model.trainable_variables)."""
    if not optimizer.built:
        raise ValueError(
            "Build the optimizer before capturing its state."
        )

    variables = tuple(optimizer.variables)
    values = []

    for variable in variables:
        value = np.array(
            keras.ops.convert_to_numpy(variable),
            copy=True,
        )
        value.setflags(write=False)
        values.append(value)

    return OptimizerSnapshot(
        optimizer=optimizer,
        variables=variables,
        values=tuple(values),
    )


def restore_optimizer_state(
    optimizer: keras.optimizers.Optimizer,
    snapshot: OptimizerSnapshot,
) -> None:
    """Restore captured variables without replacing or recompiling optimizer."""
    if optimizer is not snapshot.optimizer:
        raise ValueError(
            "Snapshot belongs to a different optimizer instance."
        )

    variables = tuple(optimizer.variables)

    if len(variables) != len(snapshot.variables):
        raise ValueError(
            "Optimizer variable layout changed. Reinitialize the model."
        )

    # Validate everything before assigning any variable.
    for index, (variable, original, value) in enumerate(
        zip(variables, snapshot.variables, snapshot.values)
    ):
        if variable is not original:
            raise ValueError(
                f"Optimizer variable {index} was replaced."
            )

        if tuple(variable.shape) != value.shape:
            raise ValueError(
                f"Optimizer variable {index} changed shape."
            )

    for variable, value in zip(variables, snapshot.values):
        optimizer.assign(variable, value)