from dataclasses import dataclass, field
from typing import Optional, Literal

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
    
    # Constraint timing
    constraint_enforcement: Literal['post_batch', 'post_epoch'] = 'post_batch'
    
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

@dataclass    
class HiperparametesFineTuningConfig:
    """
    Configuration of regularization parameters - grid search vs random search
    """
    tuning_method: Literal['grid', 'random'] = 'grid'
    epochs: int = 200
    learning_rate: float = 0.001

    hiperparameter_of: Literal['regularization', 'model'] = 'regularization'
    hiperparameter_name: str = 'alpha'
    
    n_trials: int = 20  # only for random search
    param_grid: dict = field(default_factory=dict)  # Only for grid search