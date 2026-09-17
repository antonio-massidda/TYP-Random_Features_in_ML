import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import spmm
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class MPNNConfig:
    in_channels: int
    hidden_channels_list: List[int]
    layer_type: str  # 'GCN' or 'GIN'
    use_batchnorm: bool 
    use_dropout: bool 
    freeze_weights: bool 
    pooling: str 
    activation: Optional[str] 
    use_last_activation: bool 
    weight_sampling: str  
    use_bias: bool 
    dropout_rate: float = 0.5
    weight_init_options: Optional[Dict[str, float]] = None
    
    num_models: int = 1 

def get_activation(activation: Optional[str] = None):
    activations = {
        'ReLU': F.relu,
        'LeakyReLU': F.leaky_relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh
    }
    return activations.get(activation, lambda x: x)

def apply_pooling(x, batch, pooling_type: str):
    if pooling_type == 'mean':
        return global_mean_pool(x, batch)
    elif pooling_type == 'sum':
        return global_add_pool(x, batch)
    elif pooling_type == 'max':
        return global_max_pool(x, batch)
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")

def get_init_fn(config: MPNNConfig):
    """Return the requested initializer."""
    options = config.weight_init_options or {}
    
    if config.weight_sampling == 'normal':
        mean = options.get("mean", 0.0)
        std = options.get("std", 1.0)
        return lambda t: nn.init.normal_(t, mean=mean, std=std)
    elif config.weight_sampling == 'uniform':
        min_val = options.get("min", -0.1)
        max_val = options.get("max", 0.1)
        return lambda t: nn.init.uniform_(t, a=min_val, b=max_val)
    elif config.weight_sampling == 'xavier':
        return nn.init.xavier_normal_
    elif config.weight_sampling == 'xavier_uniform':
        return nn.init.xavier_uniform_
    elif config.weight_sampling == 'orthogonal':
        return nn.init.orthogonal_
    elif config.weight_sampling == 'default':
        return None
    else:
        raise ValueError(f"Unknown weight sampling: {config.weight_sampling}")

class MPNN(nn.Module):
    def __init__(self, config: MPNNConfig):
        super(MPNN, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if config.use_batchnorm else None
        self.activation = get_activation(config.activation)

        in_dim = config.in_channels
        
        for hidden_dim in config.hidden_channels_list:
            if config.layer_type == 'GCN':
                self.layers.append(GCNConv(in_dim, hidden_dim, bias=config.use_bias))
            elif config.layer_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim, bias=config.use_bias),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=config.use_bias)
                )
                self.layers.append(GINConv(mlp))
            else:
                raise ValueError(f"Unknown layer: {config.layer_type}")
            
            if config.use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            in_dim = hidden_dim

        if config.freeze_weights:
            for p in self.parameters():
                p.requires_grad = False
                
        init_fn = get_init_fn(config)
        if init_fn:
            self.apply(lambda m: self._init_weights(m, init_fn))

    def _init_weights(self, m, init_fn):
        if isinstance(m, (nn.Linear, pyg_nn.dense.linear.Linear)):
            init_fn(m.weight)
            if m.bias is not None and self.config.use_bias:
                 nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            if i < len(self.layers) - 1 or self.config.use_last_activation:
                x = self.activation(x)
            
            if self.config.use_batchnorm:
                x = self.batch_norms[i](x)
                
            if self.config.use_dropout:
                x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
                
        return apply_pooling(x, batch, self.config.pooling)

class BatchedLinear(nn.Module):
    """Linear layer shared across a batch of models."""
    def __init__(self, num_models, in_features, out_features, bias=True, init_fn=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight = nn.Parameter(torch.empty((num_models, in_features, out_features), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty((num_models, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.init_fn = init_fn
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fn:
            self.init_fn(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)
            
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, x):
        return torch.einsum('nmi,mio->nmo', x, self.weight) + self.bias

class BatchedGINLayer(nn.Module):
    """GIN layer for the batched model."""
    def __init__(self, num_models, in_dim, hidden_dim, bias, init_fn, device, dtype):
        super().__init__()
        self.lin1 = BatchedLinear(num_models, in_dim, hidden_dim, bias=bias, init_fn=init_fn, device=device, dtype=dtype)
        self.lin2 = BatchedLinear(num_models, hidden_dim, hidden_dim, bias=bias, init_fn=init_fn, device=device, dtype=dtype)
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = nn.Parameter(torch.empty((num_models, 1, 1), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        nn.init.uniform_(self.eps, -0.2, 0.2)

    def forward(self, x, aggr):
        h = (1 + self.eps.view(1, -1, 1)) * x + aggr
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        return h

class BatchedMPNN(nn.Module):
    def __init__(self, config: MPNNConfig, device=None, dtype=None):
        super(BatchedMPNN, self).__init__()
        self.config = config
        self.num_models = config.num_models
        self.activation = get_activation(config.activation)
        
        init_fn = get_init_fn(config)
        
        self.layers = nn.ModuleList()
        in_dim = config.in_channels
        
        for hidden_dim in config.hidden_channels_list:
            if config.layer_type == 'GCN':
                self.layers.append(
                    BatchedLinear(self.num_models, in_dim, hidden_dim, bias=config.use_bias, init_fn=init_fn, device=device, dtype=dtype)
                )
            elif config.layer_type == 'GIN':
                self.layers.append(
                    BatchedGINLayer(self.num_models, in_dim, hidden_dim, bias=config.use_bias, init_fn=init_fn, device=device, dtype=dtype)
                )
            
            in_dim = hidden_dim
            
        if config.freeze_weights:
            for p in self.parameters():
                p.requires_grad = False

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t, batch):
        N = x.size(0)
        M = self.num_models
        
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, M, -1)
            
        for i, layer in enumerate(self.layers):
            if self.config.layer_type == 'GCN':
                x = layer(x)
                
                F_dim = x.size(-1)
                x_flat = x.reshape(N, -1)
                
                dtype = x.dtype
                x_flat = spmm(adj_t, x_flat.float(), reduce='mean').to(dtype)
                
                x = x_flat.view(N, M, F_dim)
                
            elif self.config.layer_type == 'GIN':
                F_dim = x.size(-1)
                x_flat = x.reshape(N, -1)
                
                dtype = x.dtype
                aggr = spmm(adj_t, x_flat.float(), reduce='add').to(dtype)
                aggr = aggr.view(N, M, F_dim)
                
                x = layer(x, aggr)

            if i < len(self.layers) - 1 or self.config.use_last_activation:
                x = self.activation(x)
                
            if self.config.use_dropout:
                x = F.dropout(x, p=self.config.dropout_rate, training=self.training)

        F_out = x.size(-1)
        x_flat = x.reshape(N, -1)
        
        if self.config.pooling == 'mean':
            out = global_mean_pool(x_flat, batch)
        elif self.config.pooling == 'sum':
            out = global_add_pool(x_flat, batch)
        elif self.config.pooling == 'max':
            out = global_max_pool(x_flat, batch)
            
        return out.view(-1, M, F_out)
