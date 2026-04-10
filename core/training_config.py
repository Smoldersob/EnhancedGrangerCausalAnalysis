from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

@dataclass
class TrainingConfig:
    """
    Konfiguracja treningu - separacja base vs restricted models
    """
    
    # Base model training
    epochs_base: int = 1000
    learning_rate_base: float = 0.001
    
    # Restricted model training (fine-tuning)
    epochs_restricted: int = 500
    lr_decay_factor: float = 0.1  # lr_restricted = lr_base * 0.1
    
    # Batch configuration
    batch_size: Optional[int] = 32  # None = full batch
    
    # Validation
    validation_split: float = 0.0

    # Optional hyperoptimization setup.
    # None -> disabled
    # 'model' -> delegate to model.hyperoptimize(...)
    # 'regularization' -> grid search over regularizer params
    hiperoptimalization_state: Optional[Literal['model', 'regularization']] = None
    hiperoptimalization_conf: Dict[str, Any] = field(default_factory=dict)
    
    
    def get_restricted_lr(self) -> float:
        """Learning rate dla fine-tuningu restricted models"""
        return self.learning_rate_base * self.lr_decay_factor
    
    def get_restricted_epochs(self) -> int:
        """Liczba epok dla fine-tuningu"""
        return self.epochs_restricted
    
    @classmethod
    def from_dict(cls, config: dict) -> 'TrainingConfig':
        """Factory z user config"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})