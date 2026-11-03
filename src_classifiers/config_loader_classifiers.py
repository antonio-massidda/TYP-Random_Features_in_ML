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

def get_rf_config_from_yaml(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract RF configuration from loaded config."""
    rf_cfg = config['rf']
    
    return {
        'n_estimators': int(rf_cfg['n_estimators']),
        'max_depth': int(rf_cfg['max_depth']),
        'min_samples_leaf': int(rf_cfg['min_samples_leaf']),

        'warmup_ratio': float(rf_cfg['warmup_ratio']),
        'n_runs': int(rf_cfg['n_runs']),
    }