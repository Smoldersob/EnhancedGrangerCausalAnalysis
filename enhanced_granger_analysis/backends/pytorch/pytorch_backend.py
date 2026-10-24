from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..base_backend import BackendStrategy
from ...core.components.constraint_spec import PreparedConstraintData
from ...core.components.regularizer_spec import PreparedRegularizerData
from ...core.exceptions import BackendNotAvailableError
from ...core.models.protocol import GrangerModelProtocol

if TYPE_CHECKING:
    from .optimizers.adapter import PyTorchOptimizerAdapter


class PyTorchBackendStrategy(BackendStrategy):
    """Construct prepared PyTorch components and model adapters."""

    name = "pytorch"

    _component_names = frozenset({
        "optimizer",
        "loss",
        "callbacks",
        "regularizer",
        "constraint",
    })

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("torch") is None:
                return False

            import torch  # noqa: F401

        except ImportError:
            return False

        return True

    def _require_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise BackendNotAvailableError(
                "PyTorch is unavailable."
            ) from exc

        return torch

    def build_optimizer(
        self,
        spec: Any = None,
        **context: Any,
    ) -> PyTorchOptimizerAdapter:
        if context:
            raise TypeError(
                "PyTorch optimizer preparation does not accept context: "
                + ", ".join(sorted(context))
            )

        self._require_torch()

        from .optimizers import resolve_optimizer

        adapter = resolve_optimizer(spec)
        self._log_component("optimizer", adapter)
        return adapter

    def build_loss(self, spec: Any = None) -> Any:
        self._require_torch()

        from .losses.resolver import resolve_loss

        loss = resolve_loss(
            "mse" if spec is None else spec
        )

        if loss is None or isinstance(loss, type) or not callable(loss):
            raise TypeError(
                "Loss resolver must return a loss instance or function."
            )

        self._log_component("loss", loss)
        return loss

    def build_callbacks(self, specs: Any = None) -> list[Any]:
        if specs is None:
            return []

        self._require_torch()

        from . import callbacks as _callbacks  # noqa: F401
        from .callbacks.base import Callback
        from .callbacks.registry import resolve_callback_spec

        if isinstance(specs, Sequence) and not isinstance(
            specs, (str, bytes)
        ):
            items = list(specs)
        else:
            items = [specs]

        callbacks = []

        for spec in items:
            resolved = resolve_callback_spec(spec)

            if resolved is None:
                continue

            if isinstance(resolved, Callback):
                callback = resolved
            else:
                callback_class, params = resolved
                callback = callback_class(**params)

            if not isinstance(callback, Callback):
                raise TypeError(
                    "Callback construction returned an invalid object."
                )

            callbacks.append(callback)

        self._log_component("callbacks", callbacks)
        return callbacks

    def build_regularizer(
        self,
        spec: Any = None,
        *,
        data: PreparedRegularizerData | None = None,
    ) -> Any:
        if spec is None:
            return None

        self._require_torch()

        from .regularizers.base import BasePyTorchRegularizer

        if isinstance(spec, BasePyTorchRegularizer):
            self._log_component("regularizer", spec)
            return spec

        from .regularizers import resolve_regularizer_spec

        resolved = resolve_regularizer_spec(spec)
        if resolved is None:
            return None

        regularizer_class, _ = resolved

        if data is None:
            raise ValueError(
                "Creating a project regularizer requires "
                "PreparedRegularizerData."
            )

        regularizer = regularizer_class(data=data)

        if not isinstance(regularizer, BasePyTorchRegularizer):
            raise TypeError(
                "Regularizer construction returned an invalid object."
            )

        self._log_component("regularizer", regularizer)
        return regularizer

    def build_constraint(
        self,
        spec: Any = None,
        *,
        data: PreparedConstraintData | None = None,
    ) -> Any:
        if spec is None:
            return None

        self._require_torch()

        from .constraints.base import BasePyTorchConstraint

        if isinstance(spec, BasePyTorchConstraint):
            self._log_component("constraint", spec)
            return spec

        from .constraints import resolve_constraint_spec

        constraint_class, params = resolve_constraint_spec(spec)

        if data is None:
            raise ValueError(
                "Creating a project constraint requires "
                "PreparedConstraintData."
            )

        constraint = constraint_class(
            data=data,
            **params,
        )

        if not isinstance(constraint, BasePyTorchConstraint):
            raise TypeError(
                "Constraint construction returned an invalid object."
            )

        self._log_component("constraint", constraint)
        return constraint

    def build_model(
        self,
        spec: Any = None,
        *,
        components: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> GrangerModelProtocol:
        self._require_torch()

        from .models.registry import (
            resolve_model_spec,
            validate_model_instance,
        )

        return self._build_registered_model(
            spec,
            resolve_spec=resolve_model_spec,
            validate_instance=validate_model_instance,
            components=components,
            config=config,
            supported_components=self._component_names,
        )