from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .input_spec import ModelInputSpec
from .training_result import TrainingResult

FloatArray: TypeAlias = NDArray[np.floating[Any]]

# Kolejność i kształty tablic określa konkretna implementacja.
# Nie obejmuje masek wejściowych ani stanu optymalizatora.
ModelWeights: TypeAlias = list[FloatArray]

CallbackSelector: TypeAlias = int | str | type
CallbackUpdates: TypeAlias = Mapping[
    CallbackSelector,
    Mapping[str, Any],
]


class GrangerModelProtocol(Protocol):
    """Backend-independent, multi-output model used by Granger analysis."""

    @property
    def backend(self) -> str:
        """Canonical library backend identifier."""
        ...

    @property
    def input_spec(self) -> ModelInputSpec:
        """Structural description used to construct this model."""
        ...
        
    @property
    def is_fitted(self) -> bool:
        """Whether the model currently contains a successful fit."""
        ...

    def rebuild(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        """Create a fresh predictor with the same structural specification.

        Reinitialize numerical parameters.
        Restore the baseline input-control state.
        Recreate the optimizer and its reset baseline.
        Discard model-owned training diagnostics.
        Set is_fitted to False.

        Do not accept or retain training observations.
        Do not change input_spec.

        Callback run preparation remains the responsibility of
        reset_components() or the default fit() behavior.
        """
        ...

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        collect_history: bool = False,
        reset_components: bool = True,
        run_name: str | None = None,
        callback_updates: CallbackUpdates | None = None,
    ) -> TrainingResult:
        """Train on explicitly supplied prepared observations.

        X has shape (n_samples, input_spec.n_features).
        y has shape (n_samples, input_spec.n_outputs).

        Validate dimensions, finite values and sample alignment.
        Do not modify caller-owned arrays.
        Do not retain training observations after the call.

        Continue from current parameters and input-control state.
        Never rebuild the predictor implicitly.

        reset_components=True resets training components, not weights
        or the current input-control state.
        """
        ...

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict using the current parameters and input-control state."""
        ...

    def get_weights(self) -> ModelWeights:
        """Return independent parameter copies.

        Available immediately after construction, also before fit().
        Excludes input-control masks and optimizer/callback state.
        """
        ...

    def set_weights(self, weights: ModelWeights) -> None:
        """Restore compatible parameters and invalidate fitted status.

        Preserve input-control, optimizer and callback state.
        """
        ...

    def omit_variables(self, variable_indices: Sequence[int]) -> None:
        """Replace the current omitted prepared-feature columns.

        Relative to the baseline established by construction/rebuild().
        Preserve parameters, structural constraints and fitted status.
        """
        ...

    def reset_components(
        self,
        *,
        reset_optimizer: bool = True,
        reset_callbacks: bool = True,
        run_name: str | None = None,
        callback_updates: CallbackUpdates | None = None,
    ) -> None:
        """Reset training components without rebuilding the predictor.

        Preserve parameters, input-control state and fitted status.
        """
        ...


class CallbackManagedModelProtocol(GrangerModelProtocol, Protocol):
    """Additional contract for models with run-managed callbacks."""

    def reset_callbacks(
        self,
        run_name: str,
        callback_updates: CallbackUpdates | None = None,
    ) -> None:
        """Prepare fresh callbacks for an independent run.

        Preserve configuration unless explicitly updated.
        Discard previous callback runtime state.

        Do not change:
        - model parameters;
        - input-control state;
        - optimizer state;
        - fitted status.

        Call between fit() calls, not during training.
        """
        ...