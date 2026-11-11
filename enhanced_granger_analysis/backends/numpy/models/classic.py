from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..constraints.base import BaseNumpyConstraint
from ..regularizers.base import BaseNumpyRegularizer

from .conditional import NumpyConditionalAdapter
from .network_parts import (
    ClassicalLinearGrangerNetwork,
    validate_feature_indices,
)
from .registry import register_model


@register_model("classical", "classical_linear")
class NumpyClassicalAdapter(NumpyConditionalAdapter):
    """Shared restricted AR baseline with cause-specific extensions.

    Constructor, fit(), callbacks, solver selection and parameter
    import/export retain the signatures of NumpyConditionalAdapter.
    """

    @property
    def base_model_role(self) -> str:
        return "restricted"

    def _build_network(
        self,
        *,
        constraint: BaseNumpyConstraint | None,
        regularizer: BaseNumpyRegularizer | None,
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
                "NumpyClassicalAdapter requires "
                "ClassicalLinearGrangerNetwork."
            )

        return model

    def _training_design(self, X: np.ndarray) -> np.ndarray:
        """Do not globally remove features needed by own-history paths."""
        return X

    def _data_gradient(
        self,
        X_centered: np.ndarray,
        y_centered: np.ndarray,
    ) -> np.ndarray:
        network = self._require_model()
        mask = network.get_controlled_slope_mask()
        weights = network.coefficient_layer.weight

        residual = (
            X_centered @ (weights * mask).T
            - y_centered
        )

        gradient = (
            2.0
            * residual.T
            @ X_centered
            / (len(X_centered) * self._n_outputs)
        )

        return gradient * mask

    @staticmethod
    def _update_bias(
        network: ClassicalLinearGrangerNetwork,
        x_mean: np.ndarray,
        y_mean: np.ndarray,
    ) -> None:
        layer = network.coefficient_layer

        if layer.bias is not None:
            bias = (
                y_mean
                - x_mean @ network.effective_weight().T
            )

            if not np.isfinite(bias).all():
                raise ValueError("Analytic bias is non-finite.")

            layer.bias[...] = bias

    def _restricted_lstsq_target(
        self,
        X_centered: np.ndarray,
        y_centered: np.ndarray,
        zero_mask: np.ndarray,
    ) -> np.ndarray:
        """Solve each equation using its actual active columns."""
        network = self._require_model()

        active_mask = (
            zero_mask
            & network.get_controlled_slope_mask()
        )

        target = np.zeros(
            (self._n_outputs, self._n_features),
            dtype=np.float64,
        )

        for output_index in range(self._n_outputs):
            active = active_mask[output_index]

            if not np.any(active):
                continue

            solution = np.linalg.lstsq(
                X_centered[:, active],
                y_centered[:, output_index],
                rcond=None,
            )[0]

            target[output_index, active] = solution

        if not np.isfinite(target).all():
            raise ValueError(
                "Least-squares solution is non-finite."
            )

        constraint = network.constraint

        if (
            constraint is not None
            and not constraint.is_satisfied(target)
        ):
            raise ValueError(
                "The least-squares solution violates the constraint. "
                "Check the zero_only_mask() contract."
            )

        return target

    def _restricted_lstsq_step(
        self,
        target: np.ndarray,
        zero_mask: np.ndarray,
        relaxation: float,
    ) -> None:
        """Apply the existing damped direct-solution update."""
        if relaxation == 0:
            return

        network = self._require_model()
        layer = network.coefficient_layer

        candidate = (
            (1.0 - relaxation) * layer.weight
            + relaxation * target
        )

        active_mask = (
            zero_mask
            & network.get_controlled_slope_mask()
        )

        # Canonical zeros, also when damping a nonzero warm start.
        candidate[~active_mask] = 0.0

        if not np.isfinite(candidate).all():
            raise ValueError(
                "Least-squares update is non-finite."
            )

        constraint = network.constraint

        if (
            constraint is not None
            and not constraint.is_satisfied(candidate)
        ):
            raise ValueError(
                "Least-squares update violates the constraint."
            )

        layer.weight[...] = candidate

    def omit_variables(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Restore the AR baseline, preserving weights and fitted status."""
        validate_feature_indices(
            variable_indices,
            n_features=self.input_spec.n_features,
        )

        self._require_model().reset_to_base()

    def add_variables(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Activate exactly these additional columns relative to AR."""
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