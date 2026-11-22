from typing import Protocol

from .context import InitializationContext, WeightArray


class Initializer(Protocol):
    """Interface for model weight initialization."""

    def __call__(
        self,
        *,
        context: InitializationContext,
    ) -> list[WeightArray]: ...
