import torch
import torch.nn as nn

class TorchFeatureMask(nn.Module):
        """Elementwise input masking with a non-trainable buffer."""

        def __init__(self, n_features: int):
            super().__init__()
            self.register_buffer(
                "mask",
                torch.ones(n_features, dtype=torch.float32),
            )

        def forward(self, x):
            return x * self.mask