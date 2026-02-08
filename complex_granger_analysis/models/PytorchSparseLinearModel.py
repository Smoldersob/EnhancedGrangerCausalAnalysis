import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

class RelationExists:
    def __init__(self, relation_table, relation_list=None):
        self.not_relation = torch.tensor(relation_table, dtype=torch.float32, device=device)
        self.forced_relations = relation_list if relation_list is not None else []
    
    def __call__(self, w) -> torch.Tensor:
        w = w.clone()
        for (i_out, indices, target_sum) in self.forced_relations:
            mask = torch.zeros_like(w[i_out], device=w.device)
            indices_tensor = torch.tensor(indices, dtype=torch.int, device=w.device)
            mask[indices_tensor] = 1.0
            current_sum = (w[i_out] * mask).sum()
            diff = current_sum - target_sum
            if torch.abs(diff) > 1e-8:
                adjust = diff / (mask.sum() + 1e-8)
                w[i_out, indices_tensor] -= adjust
        return w*self.not_relation    

class FixedNormalization(nn.Module): 
    
    def __init__(self, mean, std): 
        super(FixedNormalization, self).__init__() 
        self.register_buffer('mean', torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('std', torch.tensor(np.max([std,np.ones(std.shape)*1e-3],axis=0),dtype=torch.float32))
    
    def forward(self, x): 
        return (x - self.mean) / self.std
    
    


class SparseLinearModel(nn.Module):
    
    regularizer=None
    
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 constraint:RelationExists = None,
                 lasso:float = 0.0,
                 tol:float = 1e-15,
                 fit_intercept:bool = True,
                 batch_size=32,
                 epochs:int = 10000,
                 x_norm_mean=None,
                 x_norm_std=None,
                 y_norm_mean=None,
                 y_norm_std=None,
                 seed=None):
        
        super(SparseLinearModel,self).__init__()
        
        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            self.seed=seed
        else:
            self.seed=torch.seed()
        
        self.lasso = lasso
        self.tol = tol
        self.epochs = epochs
        self.batch_size = batch_size

        self.x_norm = FixedNormalization(x_norm_mean, x_norm_std)
        self.y_norm = FixedNormalization(y_norm_mean,y_norm_std)
        
        self.dataloader = None
        self.linear = nn.Linear(input_dim, output_dim, bias=fit_intercept)
        self.constraint = constraint
        
        nn.init.zeros_(self.linear.weight)
        if fit_intercept:
            nn.init.zeros_(self.linear.bias)

    def get_weights(self):
        return (self.linear.weight.data.clone(),self.linear.bias.data.clone())

    def set_weights(self,custom_weight,custom_bias):
        with torch.no_grad():
            self.linear.weight.copy_(custom_weight)
            self.linear.bias.copy_(custom_bias)
        return
         
    def forward(self, x):
        x = self.x_norm.forward(x)
        y = self.linear.forward(x)
        y = self.y_norm.forward(y)
        return y
    
    def apply_constraint(self):
        if self.constraint is not None:
            with torch.no_grad():
                self.linear.weight.data.copy_(self.constraint(self.linear.weight.data))
    
    def penalty(self):
        if self.lasso == 0 or self.regularizer is None:
            return 0.0
        else:
            return self.lasso * self.regularizer(self.linear.weight)
        
    def train_model(self, X_train, y_train, optimizer, callbacks=[], scheduler=None, tensoboard_writer=None, verbose=False, new_data=False):
        
        self.to(device)
        self.train()
        self.writer = tensoboard_writer
        
        x=torch.tensor(X_train, dtype=torch.float32).to(device)
        y=torch.tensor(y_train, dtype=torch.float32).to(device)

        if self.dataloader is None or new_data:
            dataset = TensorDataset(x,y)
            # Create generator with seed for reproducibility
            g = torch.Generator(device='cpu')  # Generator has to be in CPU for DataLoader
            if hasattr(self, 'seed') and self.seed is not None:
                g.manual_seed(self.seed)
            self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)

        prev_epoch_loss = -1
        epoch_loss = float('inf')

        for callback in callbacks:
            if hasattr(callback, 'on_train_begining'):
                callback.on_train_begining()

        criterion = nn.MSELoss()
        if_break=False
        for epoch in range(self.epochs):
            #Early stopping 
            if (epoch>0): 
                if(np.abs(prev_epoch_loss - epoch_loss) < self.tol):
                    break
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_begining') and callback.on_epoch_begining(prev_epoch_loss,epoch_loss,epoch) is False:
                        if_break=True
                        break
                if if_break:
                    break

            
            prev_epoch_loss = epoch_loss

            for xb, yb in self.dataloader:
                optimizer.zero_grad()
                y_pred = self.forward(xb)
                loss = criterion(y_pred, yb) + self.penalty()
                loss.backward()
                optimizer.step()
                self.apply_constraint()
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss = criterion(self.forward(x),y).item()  
            
            if self.writer:
                self.writer.add_scalar('epoch_loss', epoch_loss, epoch)
            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.6f}     ", end="\r")

            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(self)
        
        for callback in callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self)
        
        
        return self
    
    def predict(self, x):
        self.eval()  # Eval mode (ex. turn off dropout)
        with torch.no_grad():
            X=torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)  # Tansfeering data
            y_pred = self.forward(X)
        return y_pred.cpu().numpy()