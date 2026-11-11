from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ....core.models.input_spec import ModelInputSpec

from ..constraints.base import BasePyTorchConstraint
from ..regularizers.base import BasePyTorchRegularizer


class TorchFeatureMask(nn.Module):
    """Input restriction independent of coefficient parameters."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if (
            isinstance(n_features, (bool, np.bool_))
            or not isinstance(n_features, (int, np.integer))
            or n_features <= 0
        ):
            raise ValueError("n_features must be a positive integer.")
        self.n_features = int(n_features)
        self.register_buffer(
            "mask", torch.ones(self.n_features, dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.mask

    def set_mask(self, mask: Any) -> None:
        value = torch.as_tensor(
            mask, dtype=self.mask.dtype, device=self.mask.device,
        )
        if value.shape != self.mask.shape:
            raise ValueError(f"Expected mask shape {tuple(self.mask.shape)}.")
        if not torch.isfinite(value).all().item():
            raise ValueError("Mask must contain finite values.")
        if not ((value == 0) | (value == 1)).all().item():
            raise ValueError("Mask must contain only zeros and ones.")
        with torch.no_grad():
            self.mask.copy_(value)


class LinearGrangerNetwork(nn.Module):
    """Multi-output linear architecture analogous to Keras LinearGrangerNetwork.

    Owns the predictor, not its training lifecycle. The regularizer operates
    on native Linear weights: (n_outputs, n_features), independently of the
    input mask. Exported kernel shape follows Keras: (n_features, n_outputs).
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        constraint: BasePyTorchConstraint | None = None,
        regularizer: BasePyTorchRegularizer | None = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("n_features", n_features), ("n_outputs", n_outputs)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(use_bias, bool):
            raise TypeError("use_bias must be a bool.")
        if regularizer is not None and not isinstance(regularizer, BasePyTorchRegularizer):
            raise TypeError("regularizer must be a BasePyTorchRegularizer or None.")
        if constraint is not None and not isinstance(constraint, BasePyTorchConstraint):
            raise TypeError("constraint must be a BasePyTorchConstraint or None.")

        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.use_bias = use_bias
        self.regularizer = regularizer
        self.constraint = constraint
        self.variable_control_layer = TorchFeatureMask(self.n_features)
        self.coefficient_layer = nn.Linear(
            self.n_features, self.n_outputs,
            bias=self.use_bias, dtype=torch.float32,
        )
        # Match the initialization family used by Keras Dense.
        nn.init.xavier_uniform_(self.coefficient_layer.weight)
        if self.coefficient_layer.bias is not None:
            nn.init.zeros_(self.coefficient_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features:
            raise ValueError(
                f"Expected inputs with shape (n_samples, {self.n_features})."
            )
        return self.coefficient_layer(self.variable_control_layer(inputs))

    def regularization_loss(self) -> torch.Tensor:
        weights = self.coefficient_layer.weight
        if self.regularizer is None:
            return weights.new_zeros(())
        value = self.regularizer(weights)
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError("Regularizer must return a scalar torch.Tensor.")
        return value

    @torch.no_grad()
    def apply_constraints(self) -> None:
        if self.constraint is None:
            return
        weights = self.coefficient_layer.weight
        value = self.constraint(weights)
        if not isinstance(value, torch.Tensor):
            raise TypeError("Constraint must return a torch.Tensor.")
        if value.shape != weights.shape:
            raise ValueError("Constraint returned an incompatible coefficient shape.")
        if value.dtype != weights.dtype or value.device != weights.device:
            raise ValueError("Constraint must preserve coefficient dtype and device.")
        if not torch.isfinite(value).all().item():
            raise ValueError("Constraint returned non-finite coefficients.")
        weights.copy_(value)

    def get_predictor_weights(self) -> list[np.ndarray]:
        layer = self.coefficient_layer
        result = [layer.weight.detach().cpu().numpy().T.copy()]
        if layer.bias is not None:
            result.append(layer.bias.detach().cpu().numpy().copy())
        return result

    def set_predictor_weights(self, weights: list[np.ndarray]) -> None:
        layer = self.coefficient_layer
        shapes = [(self.n_features, self.n_outputs)]
        if layer.bias is not None:
            shapes.append((self.n_outputs,))
        if not isinstance(weights, list) or len(weights) != len(shapes):
            raise ValueError(f"Expected a list of {len(shapes)} parameter arrays.")

        arrays = []
        for index, (value, expected) in enumerate(zip(weights, shapes)):
            raw = np.asarray(value)
            if raw.dtype.kind not in "fiu":
                raise TypeError(f"Parameter {index} must contain real numeric values.")
            if raw.shape != expected:
                raise ValueError(
                    f"Parameter {index}: expected shape {expected}, got {raw.shape}."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.array(raw, dtype=np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"Parameter {index} contains non-finite float32 values.")
            arrays.append(array)

        # Prepare every native tensor before changing any parameter.
        parameters = [layer.weight]
        values = [arrays[0].T]
        if layer.bias is not None:
            parameters.append(layer.bias)
            values.append(arrays[1])
        tensors = [
            torch.tensor(value, dtype=parameter.dtype, device=parameter.device)
            for value, parameter in zip(values, parameters)
        ]
        if any(not torch.isfinite(value).all().item() for value in tensors):
            raise ValueError("Weights are not finite in the model's numeric dtype.")
        with torch.no_grad():
            for parameter, value in zip(parameters, tensors):
                parameter.copy_(value)


### Classical autoregressive baseline network with additional inputs.

def validate_feature_indices(
    variable_indices: Sequence[int],
    *,
    n_features: int,
) -> np.ndarray:
    """Validate column indices of the already prepared, lagged X."""
    indices = list(variable_indices)

    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise TypeError(
            "Feature indices must be integers, not booleans."
        )

    if any(index < 0 or index >= n_features for index in indices):
        raise ValueError("Feature index is out of range.")

    return np.asarray(indices, dtype=np.intp)


def _build_ar_selector(input_spec: ModelInputSpec) -> np.ndarray:
    """Select positive-lag own-history features for each output."""
    selector = np.zeros(
        (input_spec.n_outputs, input_spec.n_features),
        dtype=bool,
    )
    lags = np.asarray(input_spec.feature_lags, dtype=np.int64)

    for output_index, effect in enumerate(input_spec.effects):
        block = input_spec.cause_slices.get(effect)

        if block is None:
            raise ValueError(
                f"Output {effect!r} has no own-history block in X. "
                "The classical restricted baseline requires its "
                "lagged values among the prepared predictors."
            )

        columns = np.arange(
            block.start,
            block.stop,
            dtype=np.intp,
        )
        own_history = columns[lags[columns] > 0]

        if own_history.size == 0:
            raise ValueError(
                f"Output {effect!r} has no positive-lag "
                "own-history features in X."
            )

        selector[output_index, own_history] = True

    return selector


class ClassicalLinearGrangerNetwork(LinearGrangerNetwork):
    """Multi-output linear predictor with an autoregressive baseline.

    Owns one coefficient matrix with shape:
        (n_outputs, n_features)

    Prediction:
        AR(X, W * A) + EXT(X * m, W * E) + bias

    A and E are fixed complementary selectors.
    m is the dynamic feature-control vector.

    Constraints, regularization and predictor-weight import/export
    retain the contracts inherited from LinearGrangerNetwork.
    """

    def __init__(
        self,
        *,
        input_spec: ModelInputSpec,
        constraint: BasePyTorchConstraint | None = None,
        regularizer: BasePyTorchRegularizer | None = None,
        use_bias: bool = True,
    ) -> None:
        if not isinstance(input_spec, ModelInputSpec):
            raise TypeError("input_spec must be a ModelInputSpec.")

        ar_selector = _build_ar_selector(input_spec)

        super().__init__(
            n_features=input_spec.n_features,
            n_outputs=input_spec.n_outputs,
            constraint=constraint,
            regularizer=regularizer,
            use_bias=use_bias,
        )

        self.register_buffer(
            "ar_selector",
            torch.from_numpy(ar_selector.copy()),
        )
        self.register_buffer(
            "external_selector",
            torch.from_numpy(~ar_selector),
        )

        self.reset_to_base()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.n_features:
            raise ValueError(
                "Expected inputs with shape "
                f"(n_samples, {self.n_features})."
            )

        layer = self.coefficient_layer
        weights = layer.weight

        autoregressive = F.linear(
            inputs,
            weights * self.ar_selector,
            bias=None,
        )

        additional = F.linear(
            self.variable_control_layer(inputs),
            weights * self.external_selector,
            bias=None,
        )

        prediction = autoregressive + additional

        if layer.bias is not None:
            prediction = prediction + layer.bias

        return prediction

    @torch.no_grad()
    def reset_to_base(self) -> None:
        """Disable all additional inputs, preserving predictor weights."""
        self.variable_control_layer.mask.zero_()

    @torch.no_grad()
    def select_additional_features(
        self,
        variable_indices: Sequence[int],
    ) -> None:
        """Replace the selected additional inputs; never accumulate."""
        indices = validate_feature_indices(
            variable_indices,
            n_features=self.n_features,
        )

        mask = np.zeros(self.n_features, dtype=np.float32)
        mask[indices] = 1.0

        self.variable_control_layer.set_mask(mask)

    def get_controlled_slope_mask(self) -> np.ndarray:
        """Return architectural activity before structural constraints."""
        control = self.variable_control_layer.mask.bool()

        active = self.ar_selector | (
            self.external_selector & control.unsqueeze(0)
        )

        return active.detach().cpu().numpy().copy()