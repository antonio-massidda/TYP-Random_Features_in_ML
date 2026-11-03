#!/usr/bin/env python
"""Run the sampling experiments."""

import os
import argparse
import sys
import torch

os.environ['PYTHONUNBUFFERED'] = '1'
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import List, Dict, Any, Optional

from sampling import (
    SamplingConfig,
    ExperimentConfig,
    run_experiment,
    save_results
)
from config_loader import load_config, get_sampling_configs_from_yaml, get_rf_config_from_yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_classifiers.yaml"

# MY NOTE: get dataset name and details from the YAML file
def get_datasets_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract dataset entries from the config."""
    if 'datasets' in config:
        return config['datasets']
    if 'dataset' in config:
        return [config['dataset']]
    raise ValueError("Config must have 'dataset' or 'datasets' key")

# MY NOTE: run 80 rounds of sampling 2000 GNNs for each dataset
def run_all_experiments(
    config_path: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """Run all configured sampling experiments."""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
        config_label = config_path
    else:
        config_label = str(config_path)
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    sampling_configs = get_sampling_configs_from_yaml(config)
    rf_config = get_rf_config_from_yaml(config)
    datasets = get_datasets_from_config(config)
    total_samples = config['sampling']['total_samples']
    chunk_size = config['sampling']['chunk_size']
    output_dir = config['output']['dir']
    device = config.get('device', 'cuda:0')
    
    print("="*70)
    print("SAMPLING-BASED GNN EXPERIMENT SUITE")
    print("="*70)
    print(f"Config: {config_label}")
    print(f"Datasets: {[d['name'] for d in datasets]}")
    print(f"Candidate models sampled: {total_samples}")
    print(f"GNN configs to test: {len(sampling_configs)}")
    print(f"RF configs per GNN: {len(rf_config['n_estimators']) * len(rf_config['max_depth']) * len(rf_config['min_samples_leaf'])}")
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
        experiment_folder = Path(output_dir) / f"{dataset_name}_{timestamp}"
        experiment_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"CV Folds: {n_splits}")
        print(f"Output folder: {experiment_folder}")
        
        saved_files = run_dataset_experiments(
            dataset_name=dataset_name,
            n_splits=n_splits,
            cv_seed=cv_seed,
            sampling_configs=sampling_configs,
            rf_config=rf_config,
            total_samples=total_samples,
            chunk_size=chunk_size,
            device=device,
            experiment_folder=experiment_folder,
            verbose=verbose
        )
        
        all_saved_files.extend(saved_files)
        
        print(f"\n>>> Dataset {dataset_name} complete: {len(saved_files)} result files")
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total result files: {len(all_saved_files)}")
    
    return all_saved_files


def run_dataset_experiments(
    dataset_name: str,
    n_splits: int,
    cv_seed: int,
    sampling_configs: List[Dict[str, Any]],
    rf_config: Dict[str, Any],
    total_samples: int,
    chunk_size: int,
    device: str,
    experiment_folder: Path,
    verbose: bool = True
) -> List[str]:
    """Run all GNN config experiments for a single dataset."""
    saved_files = []
    
    for i, res_cfg in enumerate(sampling_configs):
        print(f"\n{'#'*70}")
        print(f"# GNN CONFIG {i+1}/{len(sampling_configs)}")
        print(f"# {res_cfg['layer_type']} {res_cfg['hidden_channels_list']} pooling={res_cfg['pooling']} std={res_cfg['weight_init_std']}")
        print(f"{'#'*70}")
        
        torch.cuda.empty_cache()
        
        layer_type = res_cfg['layer_type']
        layer_folder = experiment_folder / layer_type
        layer_folder.mkdir(parents=True, exist_ok=True)
        
        sampling = SamplingConfig(
            hidden_channels_list=res_cfg['hidden_channels_list'],
            layer_type=layer_type,
            pooling=res_cfg['pooling'],
            weight_init_std=res_cfg['weight_init_std'],
            total_samples=total_samples,
            chunk_size=chunk_size,
        )
        
        arch_str = (
            f"{layer_type}_{'x'.join(map(str, res_cfg['hidden_channels_list']))}_"
            f"{res_cfg['pooling']}_std{res_cfg['weight_init_std']}"
        )
        
        config = ExperimentConfig(
            dataset_name=dataset_name,
            n_splits=n_splits,
            cv_seed=cv_seed,
            device=device,
            sampling=sampling,
            rf_n_estimators=rf_config['n_estimators'], 
            rf_max_depth=rf_config['max_depth'],
            rf_min_samples_leaf=rf_config['min_samples_leaf'],
            rf_warmup_ratio=rf_config['warmup_ratio'],
            rf_n_runs=rf_config['n_runs'],
            output_dir=str(layer_folder),
            experiment_name=arch_str
        )
        
        results = run_experiment(config, verbose=verbose)
        
        filepath = save_results(results, config)
        saved_files.append(filepath)
        
        print(f"\n>>> Results saved to: {filepath}")
        print(f">>> Best val acc:       {results['best_config']['val_acc_mean']:.4f} ± {results['best_config']['val_acc_std']:.4f}")
        print(f">>> Selected test acc:  {results['best_config']['test_acc_mean']:.4f} ± {results['best_config']['test_acc_std']:.4f}")
        print(f">>> Best val_last acc:  {results['best_config_last']['val_acc_last_mean']:.4f} ± {results['best_config_last']['val_acc_last_std']:.4f}")
        print(f">>> Selected test_last: {results['best_config_last']['test_acc_last_mean']:.4f} ± {results['best_config_last']['test_acc_last_std']:.4f}")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUITE COMPLETE")
    print("="*70)
    print(f"Total experiments: {len(saved_files)}")
    print(f"Results folder: {experiment_folder}/")
    print(f"\nStructure:")
    for f in saved_files:
        rel_path = Path(f).relative_to(experiment_folder)
        print(f"  {rel_path}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Run sampling-based GNN experiments')
    parser.add_argument('--config', type=str,
                        help='Optional path to an external YAML config file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    run_all_experiments(
        config_path=args.config,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
