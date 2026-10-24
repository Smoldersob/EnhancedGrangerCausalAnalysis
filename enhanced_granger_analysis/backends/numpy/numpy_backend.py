from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any

from ..base_backend import BackendStrategy
from ...core.components.constraint_spec import PreparedConstraintData
from ...core.components.regularizer_spec import PreparedRegularizerData
from ...core.exceptions import BackendNotAvailableError
from ...core.models.protocol import GrangerModelProtocol


class NumpyBackendStrategy(BackendStrategy):
    """Construct numpy backend components and model adapters."""

    name = "numpy"

    _component_names = frozenset({
        "callbacks",
        "regularizer",
        "constraint",
    })

    _ignored_component_names = frozenset({
        "optimizer",
        "loss",
    })

    def is_available(self) -> bool:
        try:
            if importlib.util.find_spec("numpy") is None:
                return False

            import numpy  # noqa: F401

        except ImportError:
            return False

        return True

    def _require_numpy(self) -> Any:
        try:
            import numpy
        except ImportError as exc:
            raise BackendNotAvailableError(
                "numpy is unavailable."
            ) from exc

        return numpy

    def build_optimizer(
        self,
        spec: Any = None,
        **context: Any,
    ) -> None:
        if context:
            raise TypeError(
                "Numpy optimizer construction does not accept context: "
                + ", ".join(sorted(context))
            )

        from .optimizers.resolver import resolve_optimizer

        optimizer = resolve_optimizer(spec)

        if optimizer is not None:
            raise TypeError(
                "Numpy optimizer resolver must return None."
            )

        self._log_component("optimizer", optimizer)
        return None

    def build_loss(self, spec: Any = None) -> None:
        from .losses.resolver import resolve_loss

        loss = resolve_loss(spec)

        if loss is not None:
            raise TypeError(
                "Numpy loss resolver must return None."
            )

        self._log_component("loss", loss)
        return None

    def build_callbacks(self, specs: Any = None) -> list[Any]:
        if specs is None:
            return []

        self._require_numpy()

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

        self._require_numpy()

        from .regularizers.base import BaseNumpyRegularizer

        if isinstance(spec, BaseNumpyRegularizer):
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

        if not isinstance(regularizer, BaseNumpyRegularizer):
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

        self._require_numpy()

        from .constraints.base import BaseNumpyConstraint

        if isinstance(spec, BaseNumpyConstraint):
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

        if not isinstance(constraint, BaseNumpyConstraint):
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
        self._require_numpy()

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
            ignored_components=self._ignored_component_names,
        )