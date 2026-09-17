#!/usr/bin/env python
"""Run the repeated-sampling experiments."""

import argparse
import sys
import torch
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import List, Dict, Any

from sampling_multi import (
    MultiSampleConfig, 
    SamplingConfig, 
    run_multi_sample_experiment, 
    save_results
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_multi.yaml"


def get_datasets_from_config(yaml_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract dataset entries from the config."""
    if 'datasets' in yaml_config:
        return yaml_config['datasets']
    if 'dataset' in yaml_config:
        dataset = yaml_config['dataset']
        if 'n_splits' not in dataset:
            dataset['n_splits'] = yaml_config.get('n_splits', 5)
        if 'cv_seed' not in dataset:
            dataset['cv_seed'] = yaml_config.get('cv_seed', 0)
        return [dataset]
    raise ValueError("Config must have 'dataset' or 'datasets' key")


def get_sampling_configs_from_yaml(yaml_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all sampling configurations from the YAML config."""
    sampling_dict = yaml_config.get('sampling', {})
    
    if 'hidden_channels' in sampling_dict:
        hidden_channels_list = sampling_dict.get('hidden_channels', [[256, 256]])
        layer_types = sampling_dict.get('layer_types', ['GIN'])
        poolings = sampling_dict.get('poolings', ['sum'])
        weight_init_stds = sampling_dict.get('weight_init_stds', [1.0])
        
        configs = []
        for hc, lt, pool, std in product(hidden_channels_list, layer_types, poolings, weight_init_stds):
            configs.append({
                'hidden_channels_list': hc,
                'layer_type': lt,
                'pooling': pool,
                'weight_init_std': std,
            })
        return configs
    return [{
            'hidden_channels_list': sampling_dict.get('hidden_channels_list', [256, 256]),
            'layer_type': sampling_dict.get('layer_type', 'GIN'),
            'pooling': sampling_dict.get('pooling', 'sum'),
            'weight_init_std': sampling_dict.get('weight_init_std', 1.0),
    }]


def build_config_for_gnn(
    yaml_config: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    gnn_cfg: Dict[str, Any]
) -> MultiSampleConfig:
    """Build a MultiSampleConfig for a specific dataset and GNN config."""
    sampling_dict = yaml_config.get('sampling', {})
    sampling_config = SamplingConfig(
        hidden_channels_list=gnn_cfg['hidden_channels_list'],
        layer_type=gnn_cfg['layer_type'],
        pooling=gnn_cfg['pooling'],
        activation=sampling_dict.get('activation', 'ReLU'),
        use_bias=sampling_dict.get('use_bias', True),
        use_last_activation=sampling_dict.get('use_last_activation', True),
        weight_sampling=sampling_dict.get('weight_sampling', 'normal'),
        weight_init_std=gnn_cfg['weight_init_std'],
        total_samples=sampling_dict.get('total_samples', 2000),
        chunk_size=sampling_dict.get('chunk_size', 16),
        ali_k=sampling_dict.get('ali_k', 10),
        ali_block_size=sampling_dict.get('ali_block_size', 1000),
    )
    
    training_dict = yaml_config.get('training', {})
    dataset_root = yaml_config.get('dataset_root', '/tmp')
    
    config = MultiSampleConfig(
        dataset_name=dataset_cfg['name'],
        dataset_root=dataset_root,
        device=yaml_config.get('device', 'cuda:0'),
        n_splits=dataset_cfg['n_splits'],
        cv_seed=dataset_cfg['cv_seed'],
        sampling=sampling_config,
        n_mpnn_samples=training_dict.get('n_mpnn_samples', 10),
        mlp_hidden_layers=training_dict.get('mlp_hidden_layers', [[64], [64, 64]]),
        mlp_lrs=training_dict.get('mlp_lrs', [0.001, 0.0005]),
        mlp_weight_decays=training_dict.get('mlp_weight_decays', [0, 1e-4]),
        mlp_dropout_rates=training_dict.get('mlp_dropout_rates', [0.0]),
        mlp_use_scheduler=training_dict.get('mlp_use_scheduler', [True]),
        mlp_num_epochs=training_dict.get('mlp_num_epochs', 300),
        mlp_warmup_ratio=training_dict.get('mlp_warmup_ratio', 0.1),
        mlp_last_n_epochs=training_dict.get('mlp_last_n_epochs', 20),
        output_dir=yaml_config.get('output_dir', 'results'),
    )
    
    return config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run repeated-sampling GNN experiments')
    parser.add_argument('--config', type=str,
                        help='Optional path to an external YAML config file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    args = parser.parse_args()

    verbose = not args.quiet

    config_path = args.config if args.config is not None else str(DEFAULT_CONFIG_PATH)
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f"Loading config from: {config_path}")
    yaml_config = load_yaml_config(config_path)
    config_label = config_path
    
    datasets = get_datasets_from_config(yaml_config)
    gnn_configs = get_sampling_configs_from_yaml(yaml_config)
    output_base = yaml_config.get('output_dir', 'results_multi')
    
    print("="*70)
    print("REPEATED-SAMPLING GNN EXPERIMENT SUITE")
    print("="*70)
    print(f"Config: {config_label}")
    print(f"Datasets: {[d['name'] for d in datasets]}")
    print(f"GNN configs to test: {len(gnn_configs)}")
    print(f"MPNN samples per fold: {yaml_config.get('training', {}).get('n_mpnn_samples', 10)}")
    print("="*70)
    
    all_save_paths = []
    
    for dataset_idx, dataset_cfg in enumerate(datasets):
        dataset_name = dataset_cfg['name']
        
        print(f"\n{'='*70}")
        print(f"DATASET {dataset_idx+1}/{len(datasets)}: {dataset_name}")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = Path(output_base) / f"{dataset_name}_{timestamp}"
        experiment_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Output folder: {experiment_folder}")
        
        for gnn_idx, gnn_cfg in enumerate(gnn_configs):
            print(f"\n{'#'*70}")
            print(f"# GNN CONFIG {gnn_idx+1}/{len(gnn_configs)}")
            print(f"# {gnn_cfg['layer_type']} {gnn_cfg['hidden_channels_list']} pooling={gnn_cfg['pooling']} std={gnn_cfg['weight_init_std']}")
            print(f"{'#'*70}")
            
            torch.cuda.empty_cache()
            
            config = build_config_for_gnn(yaml_config, dataset_cfg, gnn_cfg)
            
            layer_folder = experiment_folder / gnn_cfg['layer_type']
            layer_folder.mkdir(parents=True, exist_ok=True)
            config.output_dir = str(layer_folder)
            
            results = run_multi_sample_experiment(config, verbose=verbose)
            
            arch_str = f"{gnn_cfg['layer_type']}_{'x'.join(map(str, gnn_cfg['hidden_channels_list']))}_{gnn_cfg['pooling']}_std{gnn_cfg['weight_init_std']}"
            config.experiment_name = arch_str
            save_path = save_results(results, config)
            all_save_paths.append(save_path)
            
            print(f"\n>>> Results saved to: {save_path}")
        
        print(f"\n>>> Dataset {dataset_name} complete: {len(gnn_configs)} GNN configs tested")
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total GNN configs: {len(gnn_configs)}")
    print(f"Total result files: {len(all_save_paths)}")
if __name__ == "__main__":
    main()
