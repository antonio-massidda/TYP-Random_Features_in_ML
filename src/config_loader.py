from typing import Dict, Any, List
from itertools import product

# MY NOTE: Simplified loading of YAML file
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# My NOTE: Sampling n times where n equals (hidden channels x layer types x poolings x weight_init_stds)
# = 80 rounds in total
def get_sampling_configs_from_yaml(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the sampling grid from a loaded config."""
    sampling_cfg = config['sampling']
    
    hidden_channels_list = sampling_cfg['hidden_channels']
    layer_types = sampling_cfg['layer_types']
    poolings = sampling_cfg['poolings']
    weight_init_stds = sampling_cfg['weight_init_stds']
    
    configs = []
    for hidden, layer_type, pooling, std in product(
        hidden_channels_list, layer_types, poolings, weight_init_stds
    ):
        configs.append({
            'hidden_channels_list': hidden,
            'layer_type': layer_type,
            'pooling': pooling,
            'weight_init_std': std,
            'total_samples': sampling_cfg['total_samples'],
            'chunk_size': sampling_cfg['chunk_size'],
        })
    return configs


# MY NOTE: epochs are used during MLP to limit how many times we can go through a dataset
def get_mlp_config_from_yaml(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MLP configuration from loaded config."""
    mlp_cfg = config['mlp']
    
    num_epochs = mlp_cfg['num_epochs']
    if isinstance(num_epochs, int):
        num_epochs = [num_epochs]
    else:
        num_epochs = [int(x) for x in num_epochs]
    
    return {
        'hidden_layers': mlp_cfg['hidden_layers'],
        'lrs': [float(x) for x in mlp_cfg['learning_rates']],
        'weight_decays': [float(x) for x in mlp_cfg['weight_decays']],
        'dropout_rates': [float(x) for x in mlp_cfg['dropout_rates']],
        'use_scheduler': mlp_cfg['use_scheduler'],
        'num_epochs': num_epochs,
        'warmup_ratio': float(mlp_cfg['warmup_ratio']),
        'n_runs': int(mlp_cfg['n_runs']),
        'last_n_epochs': int(mlp_cfg['last_n_epochs']),
    }


def get_baseline_configs_from_yaml(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the baseline grid from a loaded config."""
    gnn_cfg = config['gnn']
    training_cfg = config['training']
    
    hidden_channels_list = gnn_cfg['hidden_channels']
    layer_types = gnn_cfg['layer_types']
    poolings = gnn_cfg['poolings']
    
    mlp_hidden_layers = training_cfg['mlp_hidden_layers']
    learning_rates = [float(x) for x in training_cfg['learning_rates']]
    weight_decays = [float(x) for x in training_cfg['weight_decays']]
    dropout_rates = [float(x) for x in training_cfg['dropout_rates']]
    
    configs = []
    for hidden, layer_type, pooling, mlp_hidden, lr, wd, dropout in product(
        hidden_channels_list, layer_types, poolings, mlp_hidden_layers,
        learning_rates, weight_decays, dropout_rates
    ):
        configs.append({
            'hidden_channels_list': hidden,
            'layer_type': layer_type,
            'pooling': pooling,
            'mlp_hidden_layers': mlp_hidden,
            'learning_rate': lr,
            'weight_decay': wd,
            'dropout_rate': dropout,
            'num_epochs': int(training_cfg['num_epochs']),
            'warmup_ratio': float(training_cfg['warmup_ratio']),
            'last_n_epochs': int(training_cfg['last_n_epochs']),
            'n_runs': int(training_cfg['n_runs']),
        })
    return configs


if __name__ == "__main__":
    print("config_loader.py expects an external YAML config path when used directly.")
