from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating[Any]]

# Kolejność i kształty tablic określa konkretna implementacja.
# Nie obejmuje masek wejściowych ani stanu optymalizatora.
ModelWeights: TypeAlias = list[FloatArray]

CallbackSelector: TypeAlias = int | str | type
CallbackUpdates: TypeAlias = Mapping[
    CallbackSelector,
    Mapping[str, Any],
]


class TrainingResult(TypedDict, total=False):
    """Training diagnostics, not Granger-test results."""

    history: Mapping[str, Sequence[float]]
    metrics: Mapping[str, float]
    n_iterations: int
    stop_reason: str | None


class GrangerModelProtocol(Protocol):
    """Backend-independent, multi-output model used by Granger analysis."""

    @property
    def backend(self) -> str:
        """Canonical library backend identifier."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the model currently contains a successful fit."""
        ...

    def initialize(
        self,
        X: FloatArray,
        *,
        targets: FloatArray,
        seed: int | None = None,
    ) -> None:
        """Bind prepared data and initialize a new model state.

        X:
            Shape (n_samples, n_features).
        targets:
            Shape (n_samples, n_outputs), including single-output models.
        seed:
            Seed for initialization, where supported.

        Requirements:
        - validate data dimensions and sample alignment;
        - initialize model parameters;
        - restore the baseline input-control state;
        - discard previous training/runtime state;
        - set is_fitted to False;
        - do not generate lags, scale data or perform stationarization.

        The implementation may convert data to its native dtype/device.
        It must not modify the supplied arrays.
        """
        ...

    def fit(
        self,
        *,
        collect_history: bool = False,
        reset_components: bool = True,
        run_name: str | None = None,
        callback_updates: CallbackUpdates | None = None,
    ) -> TrainingResult:
        """Train from current model parameters and input-control state.

        reset_components=True:
            Restore the initialization baseline of training components
            and prepare fresh callbacks.

        reset_components=False:
            Do not perform library-managed component reset.
            This does not suppress the framework's normal training hooks.

        Never reset model parameters or input-control state here.
        """
        ...

    def predict(self, X: FloatArray) -> FloatArray:
        """Return predictions of shape (n_samples, n_outputs).

        Apply the current input-control state.
        Do not modify X or model/training state.

        Raise an appropriate error if the model is not fitted or the
        feature dimension is incompatible.
        """
        ...

    def get_weights(self) -> ModelWeights:
        """Return independent NumPy copies of model parameters.

        Available after initialize(), also before fit().

        Include trainable and inference-relevant numerical parameters
        required to reproduce the model's predictor.

        Exclude:
        - input-control masks;
        - optimizer state;
        - callback state;
        - bound training data.

        Ordering and shapes are implementation-specific, but stable for
        the same architecture. Arrays need not all be matrices.
        """
        ...

    def set_weights(self, weights: ModelWeights) -> None:
        """Restore parameters compatible with the initialized architecture.

        Validate all arrays before assignment.
        Do not retain references to supplied arrays.
        Do not change input-control, optimizer or callback state.

        Mark the fit as invalidated: is_fitted becomes False.
        The caller should fit() before calling predict().
        """
        ...

    def omit_variables(self, variable_indices: Sequence[int]) -> None:
        """Replace the current set of omitted input-feature columns.

        Indices refer to columns of the prepared X, not original series.

        Each call is relative to the baseline established by initialize(),
        not cumulative. An empty sequence restores that baseline.

        Requirements:
        - reject booleans, non-integers and out-of-range indices;
        - preserve model parameters;
        - preserve externally configured structural constraints;
        - apply the omission during both fit() and predict().

        This changes the current evaluation restriction without
        invalidating already fitted numerical parameters.
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
        """Reset selected training components without changing the predictor.

        Preserve model parameters, input-control state and fitted status.
        Unsupported components may raise an explicit error.
        Absent components require no action.
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