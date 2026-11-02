from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..space.base import SearchCandidate, SearchSpace


class BaseSearchStrategy(ABC):
    """Generate candidates without constructing or fitting models."""

    @abstractmethod
    def candidates(
        self,
        space: SearchSpace,
    ) -> Iterator[SearchCandidate]:
        """Generate parameter updates from a normalized space."""
        raise NotImplementedError