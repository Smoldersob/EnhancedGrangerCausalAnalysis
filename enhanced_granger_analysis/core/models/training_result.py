from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, kw_only=True)
class TrainingResult:
    """Training diagnostics, not a Granger-test result.

    An empty result means diagnostics were not requested. In particular,
    n_iterations=None does not mean that training performed zero iterations.
    """

    history: dict[str, list[float]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    n_iterations: int | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary without aliasing mutable diagnostics."""
        return {
            "history": {name: list(values) for name, values in self.history.items()},
            "metrics": dict(self.metrics),
            "n_iterations": self.n_iterations,
            "stop_reason": self.stop_reason,
        }
