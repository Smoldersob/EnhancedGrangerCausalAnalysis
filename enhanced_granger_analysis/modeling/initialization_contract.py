from typing import Protocol

from ..core.initialization.context import InitializationContext, WeightArray


class Initializer(Protocol):
    """Interface for model weight initialization."""

    def __call__(
        self,
        *,
        context: InitializationContext,
    ) -> list[WeightArray]:
        """Return weight arrays for the model."""
        ...
