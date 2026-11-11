from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..constraints.base import BasePyTorchConstraint
from ..regularizers.base import BasePyTorchRegularizer

from .conditional import PyTorchConditionalAdapter
from .network_parts import (
    ClassicalLinearGrangerNetwork,
    validate_feature_indices,
)
from .registry import register_model


@register_model("classical", "classical_linear")
class PyTorchClassicalAdapter(PyTorchConditionalAdapter):
    """Granger adapter with a shared restricted AR baseline.

    Constructor and training-method signatures are inherited unchanged
    from PyTorchConditionalAdapter.
    """

    @property
    def base_model_role(self) -> str:
        return "restricted"

    def _build_network(
        self,
        *,
        constraint: BasePyTorchConstraint | None,
        regularizer: BasePyTorchRegularizer | None,
    ) -> ClassicalLinearGrangerNetwork:
        return ClassicalLinearGrangerNetwork(
            input_spec=self.input_spec,
            constraint=constraint,
            regularizer=regularizer,
            use_bias=self.use_bias,
        )

    def _require_model(self) -> ClassicalLinearGrangerNetwork:
        model = super()._require_model()

        if not isinstance(model, ClassicalLinearGrangerNetwork):
            raise TypeError(
                "PyTorchClassicalAdapter requires "
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

        self._require_model().reset_to_base()

    def add_variables(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Activate exactly these additional features relative to AR."""
        self._require_model().select_additional_features(
            variable_indices
        )

    def get_active_mask(self) -> np.ndarray:
        network = self._require_model()
        active = network.get_controlled_slope_mask()

        if network.constraint is None:
            structural = np.ones_like(active, dtype=bool)
        else:
            structural = np.asarray(
                network.constraint.mask,
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