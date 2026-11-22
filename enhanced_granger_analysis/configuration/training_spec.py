from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class TrainingConfig:
    """Training configuration; kept separate from execution services."""

    epochs_base: int = 1000
    learning_rate_base: float = 0.001
    epochs_restricted: int = 500
    lr_decay_factor: float = 0.1
    batch_size: Optional[int] = 32
    validation_split: float = 0.0
    hiperoptimalization_state: Optional[
        Literal["model", "regularization"]
    ] = None
    hiperoptimalization_conf: Dict[str, Any] = field(default_factory=dict)

    def get_restricted_lr(self) -> float:
        return self.learning_rate_base * self.lr_decay_factor

    def get_restricted_epochs(self) -> int:
        return self.epochs_restricted

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        return cls(
            **{
                key: value
                for key, value in config.items()
                if key in cls.__annotations__
            }
        )


__all__ = ["TrainingConfig"]
