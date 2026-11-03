"""Sampling-based experiments."""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from itertools import product
from tqdm import tqdm

from torch_geometric.utils import spmm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from dataset import GraphDataset, DataSplit, PreparedData
from mpnn import BatchedMPNN, MPNNConfig
from metrics import compute_ali
from train_mlp import TrainConfig, train_mlp_averaged, AveragedTrainResult


@dataclass
class SamplingConfig:
    """Configuration for the random GNN sampling search."""
    hidden_channels_list: List[int] = field(default_factory=lambda: [512, 512])
    layer_type: str = 'GIN'
    pooling: str = 'sum'
    activation: str = 'ReLU'
    use_bias: bool = True
    use_last_activation: bool = True
    
    weight_sampling: str = 'normal'
    weight_init_std: float = 1.0
    
    total_samples: int = 2000
    chunk_size: int = 16
    
    ali_k: int = 10
    ali_block_size: int = 1000


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    dataset_name: str = 'Mutagenicity'
    dataset_root: str = '/tmp'
    
    device: str = 'cuda:0'
    
    n_splits: int = 5
    cv_seed: int = 0
    
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    rf_n_estimators: List[int] = field(default_factory=lambda: [[100]])
    rf_max_depth: List[int | None] = field(default_factory=lambda: [[None]])
    rf_min_samples_leaf: List[int] = field(default_factory=lambda: [[1]])
    
    rf_warmup_ratio: float = 0.1
    rf_n_runs: int = 10
    
    output_dir: str = 'results'
    experiment_name: Optional[str] = None

class SingleGINLayer(nn.Module):
    """Single GIN layer for feature extraction."""
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aggr: torch.Tensor) -> torch.Tensor:
        h = (1 + self.eps) * x + aggr
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        return h


class SelectedGNN(nn.Module):
    """Single selected GNN used for feature extraction."""
    def __init__(self, config: SamplingConfig, in_channels: int):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        in_dim = in_channels
        for hidden_dim in config.hidden_channels_list:
            if config.layer_type == 'GCN':
                self.layers.append(nn.Linear(in_dim, hidden_dim, bias=config.use_bias))
            elif config.layer_type == 'GIN':
                self.layers.append(SingleGINLayer(in_dim, hidden_dim, config.use_bias))
            in_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, adj_csr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if self.config.layer_type == 'GCN':
                x = layer(x)
                x = spmm(adj_csr, x, reduce='mean')
            elif self.config.layer_type == 'GIN':
                aggr = spmm(adj_csr, x, reduce='add')
                x = layer(x, aggr)
            
            if i < len(self.layers) - 1 or self.config.use_last_activation:
                if self.config.activation == 'ReLU':
                    x = F.relu(x)
                elif self.config.activation == 'LeakyReLU':
                    x = F.leaky_relu(x)
        
        if self.config.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.config.pooling == 'sum':
            return global_add_pool(x, batch)
        elif self.config.pooling == 'max':
            return global_max_pool(x, batch)

def sample_best_mpnn(
    data: PreparedData,
    train_idx: torch.Tensor,
    config: SamplingConfig,
    verbose: bool = True
) -> Dict[str, Any]:
    """Sample random MPNNs and keep the best one by ALI."""
    device = data.device
    
    mpnn_config = MPNNConfig(
        in_channels=data.num_features,
        hidden_channels_list=config.hidden_channels_list,
        layer_type=config.layer_type,
        use_batchnorm=False,
        use_dropout=False,
        freeze_weights=True,
        pooling=config.pooling,
        activation=config.activation,
        use_last_activation=config.use_last_activation,
        weight_sampling=config.weight_sampling,
        use_bias=config.use_bias,
        num_models=config.chunk_size,
        weight_init_options={'mean': 0.0, 'std': config.weight_init_std}
    )
    
    batched_model = BatchedMPNN(mpnn_config, device=device, dtype=torch.float32)
    
    train_labels = data.labels[train_idx]
    
    best_score = -float('inf')
    best_weights = None
    
    start_time = time.time()
    num_chunks = (config.total_samples + config.chunk_size - 1) // config.chunk_size
    
    pbar = tqdm(
        range(0, config.total_samples, config.chunk_size),
        desc="  Sampling",
        total=num_chunks,
        disable=not verbose,
        dynamic_ncols=True,
        mininterval=0.1,
        file=sys.stderr
    )
    
    for chunk_start in pbar:
        batched_model.reset_parameters()
        
        with torch.no_grad():
            embeddings = batched_model(data.x, data.adj_csr, data.batch_idx)
            
            train_embeddings = embeddings[train_idx]
            scores = compute_ali(
                train_embeddings, 
                train_labels, 
                k=config.ali_k, 
                block_size=config.ali_block_size, 
                mode='classification',
                metric='l2',
            )
            
            local_max, local_idx = torch.max(scores, dim=0)
            
            if local_max > best_score:
                best_score = local_max.item()
                idx = local_idx.item()
                
                best_weights = {}
                for l_idx, layer in enumerate(batched_model.layers):
                    if config.layer_type == 'GCN':
                        w = layer.weight[idx].T.float().cpu().clone()
                        b = layer.bias[idx].float().cpu().clone() if layer.bias is not None else None
                        best_weights[f'layer_{l_idx}_weight'] = w
                        best_weights[f'layer_{l_idx}_bias'] = b
                    elif config.layer_type == 'GIN':
                        best_weights[f'layer_{l_idx}_lin1_weight'] = layer.lin1.weight[idx].T.float().cpu().clone()
                        best_weights[f'layer_{l_idx}_lin1_bias'] = layer.lin1.bias[idx].float().cpu().clone()
                        best_weights[f'layer_{l_idx}_lin2_weight'] = layer.lin2.weight[idx].T.float().cpu().clone()
                        best_weights[f'layer_{l_idx}_lin2_bias'] = layer.lin2.bias[idx].float().cpu().clone()
                        best_weights[f'layer_{l_idx}_eps'] = layer.eps[idx].squeeze().float().cpu().clone()
        
        pbar.set_postfix({'best_ALI': f'{best_score:.4f}'})
    
    pbar.close()
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"  Sampling completed in {elapsed:.1f}s, best ALI = {best_score:.4f}", flush=True)
    
    return {
        'weights': best_weights,
        'ali_score': best_score,
        'sampling_time': elapsed
    }


def extract_features(
    data: PreparedData,
    selected_weights: Dict[str, torch.Tensor],
    config: SamplingConfig
) -> torch.Tensor:
    """Extract graph features with the selected model."""
    device = data.device
    
    gnn = SelectedGNN(config, data.num_features).to(device)
    
    with torch.no_grad():
        for l_idx, layer in enumerate(gnn.layers):
            if config.layer_type == 'GIN':
                layer.lin1.weight.copy_(selected_weights[f'layer_{l_idx}_lin1_weight'].to(device))
                layer.lin1.bias.copy_(selected_weights[f'layer_{l_idx}_lin1_bias'].to(device))
                layer.lin2.weight.copy_(selected_weights[f'layer_{l_idx}_lin2_weight'].to(device))
                layer.lin2.bias.copy_(selected_weights[f'layer_{l_idx}_lin2_bias'].to(device))
                layer.eps.copy_(selected_weights[f'layer_{l_idx}_eps'].to(device))
            elif config.layer_type == 'GCN':
                layer.weight.copy_(selected_weights[f'layer_{l_idx}_weight'].to(device))
                if layer.bias is not None and selected_weights[f'layer_{l_idx}_bias'] is not None:
                    layer.bias.copy_(selected_weights[f'layer_{l_idx}_bias'].to(device))
    
    gnn.eval()
    
    with torch.no_grad():
        features = gnn(data.x, data.adj_csr, data.batch_idx)
    
    return features

def run_experiment(config: ExperimentConfig, verbose: bool = True) -> Dict[str, Any]:
    """Run the full sampling pipeline."""
    device = torch.device(config.device if torch.cuda.is_available() or config.device == 'cpu' else 'cpu')
    
    if verbose:
        print("="*70)
        print("Sampling-Based GNN Experiment")
        print("="*70)
        print(f"Dataset: {config.dataset_name}")
        print(f"Device: {device}")
        print(f"CV Folds: {config.n_splits}")
        print("="*70)
    
    dataset = GraphDataset(
        name=config.dataset_name,
        root=config.dataset_root,
        device=device
    )
    data = dataset.get_prepared_data()
    
    if verbose:
        print(f"Loaded: {data.num_graphs} graphs, {data.num_features} features, {data.num_classes} classes")
    
    cv_splits = dataset.get_cross_validation_splits(n_splits=config.n_splits, seed=config.cv_seed)
    
    rf_param_grid = list(product(
        config.rf_n_estimators,
        config.rf_max_depth,
        config.rf_min_samples_leaf
    ))
    
    if verbose:
        print(f"RF configs: {len(rf_param_grid)}", flush=True)
        print("="*70, flush=True)
    
    results = {
        'config': asdict(config),
        'dataset_info': {
            'name': config.dataset_name,
            'num_graphs': data.num_graphs,
            'num_features': data.num_features,
            'num_classes': data.num_classes
        },
        'folds': [],
        'aggregated': {}
    }
    
    rf_config_results = {i: [] for i in range(len(rf_param_grid))}
    
    total_start = time.time()
    
    for fold_idx, split in enumerate(cv_splits):
        if verbose:
            print(f"\n{'='*70}", flush=True)
            print(f"FOLD {fold_idx + 1}/{config.n_splits}", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"Train: {len(split.train_idx)}, Val: {len(split.val_idx)}, Test: {len(split.test_idx)}", flush=True)
        
        torch.cuda.empty_cache()
        
        fold_result = {
            'fold': fold_idx,
            'split_sizes': {
                'train': len(split.train_idx),
                'val': len(split.val_idx),
                'test': len(split.test_idx)
            },
            'sampling': {},
            'rf_results': []
        }
        
        if verbose:
            print(f"\n--- Sampling {config.sampling.total_samples} candidate GNNs ---", flush=True)

        sampling_result = sample_best_mpnn(
            data=data,
            train_idx=split.train_idx,
            config=config.sampling,
            verbose=verbose
        )
        
        fold_result['sampling'] = {
            'ali_score': sampling_result['ali_score'],
            'sampling_time': sampling_result['sampling_time']
        }
        
        if verbose:
            print(f"\n--- Extracting features ---", flush=True)
        
        features = extract_features(
            data=data,
            selected_weights=sampling_result['weights'],
            config=config.sampling
        )
        
        if verbose:
            print(f"  Feature shape: {features.shape}", flush=True)
        
        if verbose:
            print(f"\n--- Training {len(rf_param_grid)} RF configs (x{config.mlp_n_runs} runs each) ---", flush=True)
        
        rf_pbar = tqdm(
            enumerate(rf_param_grid),
            desc="  RF Training",
            total=len(rf_param_grid),
            disable=not verbose,
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stderr,
            leave=False
        )
        
        for cfg_idx, (hidden, lr, wd, dropout, use_sched, num_epochs) in mlp_pbar:
            train_config = TrainConfig(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,

                warmup_ratio=config.mlp_warmup_ratio,
            )
            
            rf_result = train_rf_averaged(
                features=features,
                labels=data.labels,
                train_idx=split.train_idx,
                val_idx=split.val_idx,
                test_idx=split.test_idx,
                config=train_config,
                n_runs=config.rf_n_runs,
                device=device,
                verbose=False
            )
            
            result_dict = {
                'config_idx': cfg_idx,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,

                'val_acc': rf_result.val_acc,
                'val_acc_std': rf_result.val_acc_std,
                'val_acc_last': rf_result.val_acc_last,
                'val_acc_last_std': rf_result.val_acc_last_std,
                'test_acc': rf_result.test_acc,
                'test_acc_std': rf_result.test_acc_std,
                'test_acc_last': rf_result.test_acc_last,
                'test_acc_last_std': rf_result.test_acc_last_std,
                'best_epoch': rf_result.best_epoch
            }
            
            fold_result['rf_results'].append(result_dict)
            rf_config_results[cfg_idx].append(result_dict)
            
            rf_pbar.set_postfix({
                'Test': f'{rf_result.test_acc:.3f}',
                'Val': f'{rf_result.val_acc:.3f}'
            })
        
        rf_pbar.close()
        
        if verbose:
            print(f"\n  {'Config':<35} {'Val':>12} {'Test':>12} {'TestLast':>12}")
            print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
            for res in fold_result['rf_results']:
                config_str = f"trees={res['n_estimators']} max_depth={res['max_depth']} min_samples_leaf={res['min_samples_leaf']}"
                print(f"  {config_str:<35} "
                      f"{res['val_acc']:.4f}±{res['val_acc_std']:.3f} "
                      f"{res['test_acc']:.4f}±{res['test_acc_std']:.3f} "
                      f"{res['test_acc_last']:.4f}±{res['test_acc_last_std']:.3f}")
        
        results['folds'].append(fold_result)
    
    if verbose:
        print(f"\n{'='*70}")
        print("AGGREGATING RESULTS ACROSS FOLDS")
        print("="*70)
    
    aggregated = []
    for cfg_idx, (n_estimators, max_depth, min_samples_leaf) in enumerate(rf_param_grid):
        fold_results = rf_config_results[cfg_idx]

        test_accs = [r['test_acc'] for r in fold_results]
        test_accs_last = [r['test_acc_last'] for r in fold_results]
        val_accs = [r['val_acc'] for r in fold_results]
        val_accs_last = [r['val_acc_last'] for r in fold_results]
        
        agg = {
            'config_idx': cfg_idx,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'test_acc_mean': float(np.mean(test_accs)),
            'test_acc_std': float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
            'test_acc_last_mean': float(np.mean(test_accs_last)),
            'test_acc_last_std': float(np.std(test_accs_last, ddof=1)) if len(test_accs_last) > 1 else 0.0,
            'val_acc_mean': float(np.mean(val_accs)),
            'val_acc_std': float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0,
            'val_acc_last_mean': float(np.mean(val_accs_last)),
            'val_acc_last_std': float(np.std(val_accs_last, ddof=1)) if len(val_accs_last) > 1 else 0.0,
            'test_accs_per_fold': test_accs,
            'test_accs_last_per_fold': test_accs_last,
            'val_accs_per_fold': val_accs,
            'val_accs_last_per_fold': val_accs_last
        }
        aggregated.append(agg)

    aggregated.sort(key=lambda x: x['val_acc_mean'], reverse=True)
    results['aggregated'] = aggregated

    best = aggregated[0]
    results['best_config'] = {
        'n_estimators': best['n_estimators'],
        'max_depth': best['max_depth'],
        'min_samples_leaf': best['min_samples_leaf'],

        'val_acc_mean': best['val_acc_mean'],
        'val_acc_std': best['val_acc_std'],
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_std': best['test_acc_std'],
        'selection_metric': 'val_acc_mean'
    }

    aggregated_by_last = sorted(aggregated, key=lambda x: x['val_acc_last_mean'], reverse=True)
    best_last = aggregated_by_last[0]
    results['best_config_last'] = {
        'n_estimators': best_last['n_estimators'],
        'max_depth': best_last['max_depth'],
        'min_samples_leaf': best_last['min_samples_leaf'],

        'val_acc_last_mean': best_last['val_acc_last_mean'],
        'val_acc_last_std': best_last['val_acc_last_std'],
        'test_acc_last_mean': best_last['test_acc_last_mean'],
        'test_acc_last_std': best_last['test_acc_last_std'],
        'selection_metric': 'val_acc_last_mean'
    }
    
    total_time = time.time() - total_start
    results['total_time_seconds'] = total_time
    
    if verbose:
        print(f"\n--- Top 5 RF Configurations (by mean VAL acc across folds) ---")
        for i, agg in enumerate(aggregated[:5]):
            print(f"  {i+1}. trees={agg['n_estimators']} max_depth={agg['max_depth']} min_samples_leaf={agg['min_samples_leaf']} | "
                  f"Val: {agg['val_acc_mean']:.4f}±{agg['val_acc_std']:.4f} | "
                  f"Test: {agg['test_acc_mean']:.4f}±{agg['test_acc_std']:.4f}")

        print(f"\n--- Top 5 RF Configurations (by mean VAL_LAST acc across folds) ---")
        for i, agg in enumerate(aggregated_by_last[:5]):
            print(f"  {i+1}. trees={agg['n_estimators']} max_depth={agg['max_depth']} min_samples_leaf={agg['min_samples_leaf']} | "
                  f"ValLast: {agg['val_acc_last_mean']:.4f}±{agg['val_acc_last_std']:.4f} | "
                  f"TestLast: {agg['test_acc_last_mean']:.4f}±{agg['test_acc_last_std']:.4f}")

        print(f"\n{'='*70}")
        print(f"BEST CONFIG (by Val): {best['n_estimators']}, max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}")
        print(f"  Val Accuracy:  {best['val_acc_mean']:.4f} ± {best['val_acc_std']:.4f}")
        print(f"  Test Accuracy: {best['test_acc_mean']:.4f} ± {best['test_acc_std']:.4f}")
        print(f"\nBEST CONFIG (by ValLast): {best_last['n_estimators']}, max_depth={best_last['max_depth']}, min_samples_leaf={best_last['min_samples_leaf']}")
        print(f"  ValLast Accuracy:  {best_last['val_acc_last_mean']:.4f} ± {best_last['val_acc_last_std']:.4f}")
        print(f"  TestLast Accuracy: {best_last['test_acc_last_mean']:.4f} ± {best_last['test_acc_last_std']:.4f}")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} minutes")
    
    return results


def save_results(results: Dict[str, Any], config: ExperimentConfig) -> str:
    """Save results to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.experiment_name:
        filename = f"{config.experiment_name}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sampling_{config.dataset_name}_{timestamp}.json"
    
    filepath = output_dir / filename
    
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    results = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(filepath)
