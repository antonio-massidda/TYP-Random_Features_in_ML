#!/usr/bin/env python
"""Run the end-to-end baseline experiments."""

import os
import argparse
import sys
import torch

os.environ['PYTHONUNBUFFERED'] = '1'
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import List, Dict, Any, Optional

from baseline import (
    BaselineConfig,
    run_baseline_experiment,
    save_baseline_results
)
from config_loader import load_config, get_baseline_configs_from_yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_baseline.yaml"


def get_datasets_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract dataset entries from the config."""
    if 'datasets' in config:
        return config['datasets']
    if 'dataset' in config:
        return [config['dataset']]
    raise ValueError("Config must have 'dataset' or 'datasets' key")


def run_all_baseline_experiments(
    config_path: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """Run all configured baseline experiments."""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
        config_label = config_path
    else:
        config_label = str(config_path)
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    baseline_configs = get_baseline_configs_from_yaml(config)
    datasets = get_datasets_from_config(config)
    output_dir = config['output']['dir']
    device = config.get('device', 'cuda:0')
    
    gnn_groups = {}
    for cfg in baseline_configs:
        key = (cfg['layer_type'], tuple(cfg['hidden_channels_list']), cfg['pooling'])
        if key not in gnn_groups:
            gnn_groups[key] = []
        gnn_groups[key].append(cfg)
    
    print("="*70)
    print("BASELINE END-TO-END GNN EXPERIMENT SUITE")
    print("="*70)
    print(f"Config: {config_label}")
    print(f"Datasets: {[d['name'] for d in datasets]}")
    print(f"GNN architectures to test: {len(gnn_groups)}")
    print(f"Total training configs: {len(baseline_configs)}")
    print("="*70)
    
    all_saved_files = []
    
    for dataset_idx, dataset_cfg in enumerate(datasets):
        dataset_name = dataset_cfg['name']
        n_splits = dataset_cfg['n_splits']
        cv_seed = dataset_cfg['cv_seed']
        
        print(f"\n{'#'*70}")
        print(f"# DATASET {dataset_idx+1}/{len(datasets)}: {dataset_name}")
        print(f"{'#'*70}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_folder = Path(output_dir) / f"baseline_{dataset_name}_{timestamp}"
        experiment_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"CV Folds: {n_splits}")
        print(f"Output folder: {experiment_folder}")
        
        saved_files = run_dataset_baseline_experiments(
            dataset_name=dataset_name,
            n_splits=n_splits,
            cv_seed=cv_seed,
            gnn_groups=gnn_groups,
            device=device,
            experiment_folder=experiment_folder,
            verbose=verbose
        )
        
        all_saved_files.extend(saved_files)
        
        print(f"\n>>> Dataset {dataset_name} complete: {len(saved_files)} result files")
    
    print(f"\n{'='*70}")
    print("ALL BASELINE EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total result files: {len(all_saved_files)}")
    
    return all_saved_files


def run_dataset_baseline_experiments(
    dataset_name: str,
    n_splits: int,
    cv_seed: int,
    gnn_groups: Dict[tuple, List[Dict[str, Any]]],
    device: str,
    experiment_folder: Path,
    verbose: bool = True
) -> List[str]:
    """Run all GNN config experiments for a single dataset."""
    saved_files = []
    
    for gnn_idx, ((layer_type, hidden_channels, pooling), mlp_configs) in enumerate(gnn_groups.items()):
        print(f"\n{'#'*70}")
        print(f"# GNN ARCHITECTURE {gnn_idx+1}/{len(gnn_groups)}")
        print(f"# {layer_type} {list(hidden_channels)} pooling={pooling}")
        print(f"# ({len(mlp_configs)} MLP/training configs)")
        print(f"{'#'*70}")
        
        torch.cuda.empty_cache()
        
        layer_folder = experiment_folder / layer_type
        layer_folder.mkdir(parents=True, exist_ok=True)
        
        mlp_hidden_layers = list(set(tuple(c['mlp_hidden_layers']) for c in mlp_configs))
        mlp_hidden_layers = [list(h) for h in mlp_hidden_layers]
        lrs = list(set(c['learning_rate'] for c in mlp_configs))
        weight_decays = list(set(c['weight_decay'] for c in mlp_configs))
        dropout_rates = list(set(c['dropout_rate'] for c in mlp_configs))
        
        first_cfg = mlp_configs[0]
        
        baseline_cfg = BaselineConfig(
            dataset_name=dataset_name,
            n_splits=n_splits,
            cv_seed=cv_seed,
            device=device,
            hidden_channels_list=list(hidden_channels),
            layer_type=layer_type,
            pooling=pooling,
            mlp_hidden_layers=mlp_hidden_layers,
            lrs=lrs,
            weight_decays=weight_decays,
            mlp_dropout_rates=dropout_rates,
            use_scheduler=[True],
            num_epochs=first_cfg['num_epochs'],
            warmup_ratio=first_cfg['warmup_ratio'],
            last_n_epochs=first_cfg['last_n_epochs'],
            n_runs=first_cfg['n_runs'],
            output_dir=str(layer_folder),
        )
        
        results = run_baseline_experiment(baseline_cfg, verbose=verbose)
        
        arch_str = f"{layer_type}_{'x'.join(map(str, hidden_channels))}_{pooling}"
        baseline_cfg.experiment_name = arch_str
        
        filepath = save_baseline_results(results, baseline_cfg)
        saved_files.append(filepath)
        
        print(f"\n>>> Results saved to: {filepath}")
        print(f">>> Best val acc:       {results['best_config']['val_acc_mean']:.4f} ± {results['best_config']['val_acc_std']:.4f}")
        print(f">>> Selected test acc:  {results['best_config']['test_acc_mean']:.4f} ± {results['best_config']['test_acc_std']:.4f}")
        print(f">>> Best val_last acc:  {results['best_config_last']['val_acc_last_mean']:.4f} ± {results['best_config_last']['val_acc_last_std']:.4f}")
        print(f">>> Selected test_last: {results['best_config_last']['test_acc_last_mean']:.4f} ± {results['best_config_last']['test_acc_last_std']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"DATASET {dataset_name} COMPLETE")
    print("="*70)
    print(f"Total experiments: {len(saved_files)}")
    print(f"Results folder: {experiment_folder}/")
    print(f"\nStructure:")
    for f in saved_files:
        rel_path = Path(f).relative_to(experiment_folder)
        print(f"  {rel_path}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Run baseline end-to-end GNN experiments')
    parser.add_argument('--config', type=str,
                        help='Optional path to an external YAML config file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    run_all_baseline_experiments(
        config_path=args.config,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
