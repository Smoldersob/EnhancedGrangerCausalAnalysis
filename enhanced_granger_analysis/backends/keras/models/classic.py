from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .conditional import KerasConditionalAdapter
from .network_parts import (
    ClassicalLinearGrangerNetwork,
    build_ar_selector,
    validate_feature_indices,
)
from .registry import register_model


@register_model("classical", "classical_linear")
class KerasClassicalAdapter(KerasConditionalAdapter):
    """Granger adapter with a shared restricted AR baseline.

    Constructor and training-method signatures are inherited unchanged
    from KerasConditionalAdapter.
    """

    @property
    def base_model_role(self) -> str:
        return "restricted"

    def _build_network(
        self,
        *,
        constraint: Any,
        regularizer: Any,
    ) -> ClassicalLinearGrangerNetwork:
        return ClassicalLinearGrangerNetwork(
            n_features=self.input_spec.n_features,
            n_outputs=self.input_spec.n_outputs,
            ar_selector=build_ar_selector(self.input_spec),
            constraint=constraint,
            regularizer=regularizer,
            use_bias=self.use_bias,
            dtype="float32",
        )

    def _require_model(self) -> ClassicalLinearGrangerNetwork:
        model = super()._require_model()

        if not isinstance(model, ClassicalLinearGrangerNetwork):
            raise TypeError(
                "KerasClassicalAdapter requires "
                "ClassicalLinearGrangerNetwork."
            )

        return model

    def omit_variables(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Restore the shared AR baseline.

        Indices are validated for protocol compatibility.
        All additional inputs are disabled regardless of their values.
        Predictor weights and fitted status are preserved.
        """
        validate_feature_indices(
            variable_indices,
            n_features=self.input_spec.n_features,
        )

        with self.device_manager.scope():
            self._require_model().reset_to_base()

    def add_variables(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Activate exactly these additional features relative to AR."""
        with self.device_manager.scope():
            self._require_model().select_additional_features(
                variable_indices
            )

    def get_active_mask(self) -> np.ndarray:
        network = self._require_model()

        with self.device_manager.scope():
            active = network.get_controlled_slope_mask()

        constraint = (
            network.coefficient_layer.kernel_constraint
        )

        if constraint is None:
            structural = np.ones_like(active, dtype=bool)
        else:
            structural = np.asarray(
                constraint.mask,
                dtype=bool,
            )

            if structural.shape != active.shape:
                raise ValueError(
                    "Constraint mask has shape "
                    f"{structural.shape}; expected {active.shape}."
                )

        return np.array(
            structural & active,
            dtype=bool,
            copy=True,
        )