import np
import torch
import torch.nn as nn

class CyclicL1Regularizer(nn.Module):
    """
    L1 regularizer with cyclingly growing weights.
    Sum of |w| is multiply by coefficient from coeffs wecto.
    Cycling: indexrs of weights are are looped ever period of elements form coeffs
    vector.
    Regularizer is dedicated for models, where inputs from one period are of simiular
    nature but are order by probability of effect (ex. autoregression, power series). 
    """

    def __init__(self, coeffs=None, enable_cyclic=False):
        super().__init__()
        if coeffs is None:
            self.register_buffer('coeffs', torch.linspace(1.0, 3.0, 20))
        else:
            self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float32))
        self.enable_cyclic = enable_cyclic
        self.lag_order = 1
        
    def set_lag_orders(self,lag_order):
        self.indices=np.concat([np.arange(l) for l in np.diff(lag_order)])

    def forward(self, param):
        if not self.training or param.numel() == 0:
            return torch.tensor(0.0, device=param.device, dtype=param.dtype)
        
        abs_param = torch.abs(param)
        
        # Suma |w| per zmienna wejściowa (input features)
        if param.dim() == 1:
            weights_per_input = abs_param
        elif param.dim() == 2:  # out_features x in_features
            weights_per_input = abs_param.sum(dim=0)
        else:
            raise ValueError(f"Nieobsługiwany kształt: {param.shape}")
        
        m = len(weights_per_input)
        device = weights_per_input.device
        
        if self.enable_cyclic:
            multipliers = self.coeffs[self.indices].to(device)
        else:
            multipliers = torch.ones(m, device=device, dtype=weights_per_input.dtype)
        
        return torch.sum(weights_per_input * multipliers)
