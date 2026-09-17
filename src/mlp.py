import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_layers: List[int]
    dropout_rate: float = 0.5
    use_batchnorm: bool = True
    activation: str = 'ReLU'

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if config.use_batchnorm else None
        
        if config.activation == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()

        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
            
        self.out = nn.Linear(prev_dim, config.num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.config.use_batchnorm:
                x = self.bns[i](x)
            
            x = self.act(x)
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
            
        return self.out(x)
