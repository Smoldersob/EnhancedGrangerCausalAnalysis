from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

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
