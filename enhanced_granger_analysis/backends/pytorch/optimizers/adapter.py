from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from inspect import isabstract
from typing import Any

from torch import Tensor
from torch.optim import Optimizer


class PyTorchOptimizerAdapter:
    """Construct and reset an optimizer bound to one model.

    A new initialize() call creates a new native optimizer and establishes
    a new initial state.

    Supports one parameter group supplied as an iterable of tensors.
    """

    def __init__(
        self,
        optimizer_class: type[Optimizer],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(optimizer_class, type)
            or not issubclass(optimizer_class, Optimizer)
            or optimizer_class is Optimizer
            or isabstract(optimizer_class)
        ):
            raise TypeError(
                "optimizer_class must be a concrete "
                "torch.optim.Optimizer subclass."
            )

        options = {} if options is None else dict(options)

        if any(not isinstance(key, str) for key in options):
            raise TypeError(
                "Optimizer option names must be strings."
            )

        if "params" in options:
            raise ValueError(
                "Model parameters must be supplied to initialize(), "
                "not through optimizer options."
            )

        self._optimizer_class = optimizer_class
        self._options = deepcopy(options)

        self._native: Optimizer | None = None
        self._initial_state: dict[str, Any] | None = None
        self._parameter_ids: tuple[int, ...] = ()

    @property
    def optimizer_class(self) -> type[Optimizer]:
        return self._optimizer_class

    @property
    def options(self) -> dict[str, Any]:
        return deepcopy(self._options)

    @property
    def is_initialized(self) -> bool:
        return self._native is not None

    @property
    def native(self) -> Optimizer:
        if self._native is None:
            raise RuntimeError(
                "Optimizer is not initialized. "
                "Call initialize(parameters) first."
            )

        return self._native

    def initialize(
        self,
        parameters: Iterable[Tensor],
    ) -> Optimizer:
        parameters = list(parameters)

        if not parameters:
            raise ValueError(
                "Optimizer requires at least one parameter."
            )

        if any(
            not isinstance(parameter, Tensor)
            for parameter in parameters
        ):
            raise TypeError(
                "Expected an iterable of parameter tensors."
            )

        parameter_ids = tuple(id(p) for p in parameters)

        if len(set(parameter_ids)) != len(parameter_ids):
            raise ValueError(
                "Optimizer parameters must not contain duplicates."
            )

        optimizer = self._optimizer_class(
            parameters,
            **deepcopy(self._options),
        )

        initial_state = deepcopy(optimizer.state_dict())

        # Commit only after successful construction and snapshot creation.
        self._native = optimizer
        self._initial_state = initial_state
        self._parameter_ids = parameter_ids

        return optimizer

    def reset(self) -> None:
        """Restore optimizer state and clear gradients, preserving weights."""
        optimizer = self.native

        if self._initial_state is None:
            raise RuntimeError(
                "Initial optimizer state is unavailable."
            )

        current_ids = tuple(
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )

        if (
            len(optimizer.param_groups) != 1
            or current_ids != self._parameter_ids
        ):
            raise RuntimeError(
                "Optimizer parameter groups changed after initialization. "
                "Reinitialize the adapter explicitly."
            )

        optimizer.load_state_dict(
            deepcopy(self._initial_state)
        )
        optimizer.zero_grad(set_to_none=True)