from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.exceptions import DataValidationError, RegistryError
from ..core.initialization import InitializationContext
from ..core.models.protocol import GrangerModelProtocol
from ..initializers.registry import resolve_initializer_spec
from ..preprocessing.prepare_data import PreparedData


class ModelInitializer:
    """Compute initial weights using the existing initializer registry.

    Weight assignment and execution-state management remain the
    responsibility of the caller.
    """

    def compute_weights(
        self,
        *,
        model: GrangerModelProtocol,
        prepared: PreparedData,
        initializer_spec: Any = None,
    ) -> Sequence[np.ndarray] | None:
        """Return initial weights without assigning them to the model.

        None disables additional initialization.
        Specification validation is delegated to the existing resolver.
        """

        try:
            initializer_class, params = resolve_initializer_spec(
                initializer_spec
            )
        except (RegistryError, TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Invalid initializer specification: {exc}"
            ) from exc

        if initializer_class is None:
            return None

        initializer = initializer_class(**params)

        context = InitializationContext(
            X=prepared.X_scaled,
            y=prepared.y_scaled,
            weights=tuple(
                np.asarray(weight).copy()
                for weight in model.get_weights()
            ),
            mask=prepared.base_mask,
        )

        return initializer(context)