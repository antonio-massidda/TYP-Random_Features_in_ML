"""Repeated-sampling experiments."""

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
from typing import List, Dict, Any, Optional, Tuple
from itertools import product
from tqdm import tqdm

from torch_geometric.utils import spmm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from dataset import GraphDataset, DataSplit, PreparedData
from mpnn import BatchedMPNN, MPNNConfig
from metrics import compute_ali
from train_mlp import TrainConfig, train_mlp, TrainResult
from mlp import MLP, MLPConfig


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
class MultiSampleConfig:
    """Full experiment configuration for multi-sample approach."""
    dataset_name: str = 'Mutagenicity'
    dataset_root: str = '/tmp'
    
    device: str = 'cuda:0'
    
    n_splits: int = 5
    cv_seed: int = 0
    
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    n_mpnn_samples: int = 10
    
    mlp_hidden_layers: List[List[int]] = field(default_factory=lambda: [[64], [64, 64]])
    mlp_lrs: List[float] = field(default_factory=lambda: [0.0005, 0.001])
    mlp_weight_decays: List[float] = field(default_factory=lambda: [0, 1e-4])
    mlp_dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1])
    mlp_use_scheduler: List[bool] = field(default_factory=lambda: [True])
    
    mlp_num_epochs: int = 300
    mlp_warmup_ratio: float = 0.1
    mlp_last_n_epochs: int = 20
    
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
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Sample many random MPNNs and return the best one by ALI."""
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
    
    for chunk_start in range(0, config.total_samples, config.chunk_size):
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
    
    return best_weights, best_score


def extract_features(
    data: PreparedData,
    selected_weights: Dict[str, torch.Tensor],
    config: SamplingConfig
) -> torch.Tensor:
    """Extract features using weights from the selected model."""
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

def run_multi_sample_experiment(config: MultiSampleConfig, verbose: bool = True) -> Dict[str, Any]:
    """Run the repeated-sampling pipeline."""
    device = torch.device(config.device if torch.cuda.is_available() or config.device == 'cpu' else 'cpu')
    
    if verbose:
        print("="*70)
        print("Repeated-Sampling GNN Experiment")
        print("="*70)
        print(f"Dataset: {config.dataset_name}")
        print(f"Device: {device}")
        print(f"CV Folds: {config.n_splits}")
        print(f"MPNN Samples per fold: {config.n_mpnn_samples}")
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
    
    mlp_param_grid = list(product(
        config.mlp_hidden_layers,
        config.mlp_lrs,
        config.mlp_weight_decays,
        config.mlp_dropout_rates,
        config.mlp_use_scheduler
    ))
    
    if verbose:
        print(f"MLP configs: {len(mlp_param_grid)}")
        print("="*70)
    
    results = {
        'config': asdict(config),
        'dataset_info': {
            'name': config.dataset_name,
            'num_graphs': data.num_graphs,
            'num_features': data.num_features,
            'num_classes': data.num_classes
        },
        'folds': [],
        'all_ali_scores': [],
    }
    
    total_start = time.time()
    
    for fold_idx, split in enumerate(cv_splits):
        if verbose:
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx + 1}/{config.n_splits}")
            print(f"{'='*70}")
            print(f"Train: {len(split.train_idx)}, Val: {len(split.val_idx)}, Test: {len(split.test_idx)}")
        
        torch.cuda.empty_cache()
        
        fold_result = {
            'fold': fold_idx,
            'split_sizes': {
                'train': len(split.train_idx),
                'val': len(split.val_idx),
                'test': len(split.test_idx)
            },
            'mpnn_samples': [],
        }
        
        sample_pbar = tqdm(
            range(config.n_mpnn_samples),
            desc=f"  Fold {fold_idx+1}",
            disable=not verbose,
            dynamic_ncols=True,
            file=sys.stderr
        )
        
        for sample_idx in sample_pbar:
            sample_pbar.set_postfix({'phase': 'sampling MPNNs'})
            
            weights, ali_score = sample_best_mpnn(
                data=data,
                train_idx=split.train_idx,
                config=config.sampling
            )
            
            results['all_ali_scores'].append({
                'fold': fold_idx,
                'sample': sample_idx,
                'ali_score': ali_score
            })
            
            sample_pbar.set_postfix({'phase': 'extracting', 'ALI': f'{ali_score:.4f}'})
            
            features = extract_features(
                data=data,
                selected_weights=weights,
                config=config.sampling
            )
            
            sample_result = {
                'sample_idx': sample_idx,
                'ali_score': ali_score,
                'mlp_results': []
            }
            
            sample_pbar.set_postfix({'phase': 'training MLPs', 'ALI': f'{ali_score:.4f}'})
            
            for cfg_idx, (hidden, lr, wd, dropout, use_sched) in enumerate(mlp_param_grid):
                train_config = TrainConfig(
                    hidden_layers=hidden,
                    lr=lr,
                    weight_decay=wd,
                    dropout_rate=dropout,
                    use_scheduler=use_sched,
                    num_epochs=config.mlp_num_epochs,
                    warmup_ratio=config.mlp_warmup_ratio,
                    last_n_epochs=config.mlp_last_n_epochs
                )
                
                mlp_result = train_mlp(
                    features=features,
                    labels=data.labels,
                    train_idx=split.train_idx,
                    val_idx=split.val_idx,
                    test_idx=split.test_idx,
                    config=train_config,
                    device=device,
                    verbose=False
                )
                
                result_dict = {
                    'config_idx': cfg_idx,
                    'hidden_layers': hidden,
                    'lr': lr,
                    'weight_decay': wd,
                    'dropout_rate': dropout,
                    'use_scheduler': use_sched,
                    'val_acc': mlp_result.val_acc,
                    'val_acc_last': mlp_result.val_acc_last,
                    'test_acc': mlp_result.test_acc,
                    'test_acc_last': mlp_result.test_acc_last,
                    'best_epoch': mlp_result.best_epoch,
                }
                
                sample_result['mlp_results'].append(result_dict)
            
            fold_result['mpnn_samples'].append(sample_result)
            
            best_test = max(r['test_acc'] for r in sample_result['mlp_results'])
            sample_pbar.set_postfix({
                'ALI': f'{ali_score:.4f}',
                'Test': f'{best_test:.4f}',
            })
        
        sample_pbar.close()
        results['folds'].append(fold_result)
        
        if verbose:
            fold_ali_scores = [s['ali_score'] for s in fold_result['mpnn_samples']]
            print(f"\n  ALI scores: mean={np.mean(fold_ali_scores):.4f}, "
                  f"std={np.std(fold_ali_scores):.4f}, "
                  f"min={np.min(fold_ali_scores):.4f}, max={np.max(fold_ali_scores):.4f}")
            
            for sample in fold_result['mpnn_samples']:
                print(f"\n  --- MPNN Sample {sample['sample_idx']+1} (ALI={sample['ali_score']:.4f}) ---")
                print(f"  {'Config':<25} {'Val':>8} {'Test':>8} {'TestLast':>8}")
                print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
                
                for res in sample['mlp_results']:
                    h_str = str(res['hidden_layers'])
                    config_str = f"h={h_str:<8} lr={res['lr']:.4f}"
                    print(f"  {config_str:<25} "
                          f"{res['val_acc']:.4f}   "
                          f"{res['test_acc']:.4f}   "
                          f"{res['test_acc_last']:.4f}")
            
            print(f"\n  === FOLD {fold_idx+1} AVERAGE (across {config.n_mpnn_samples} MPNN samples) ===")
            print(f"  {'Config':<25} {'Val':>12} {'Test':>12} {'TestLast':>12}")
            print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
            
            for cfg_idx, (hidden, lr, wd, dropout, use_sched) in enumerate(mlp_param_grid):
                cfg_vals = [s['mlp_results'][cfg_idx]['val_acc'] for s in fold_result['mpnn_samples']]
                cfg_tests = [s['mlp_results'][cfg_idx]['test_acc'] for s in fold_result['mpnn_samples']]
                cfg_lasts = [s['mlp_results'][cfg_idx]['test_acc_last'] for s in fold_result['mpnn_samples']]
                
                h_str = str(hidden)
                config_str = f"h={h_str:<8} lr={lr:.4f}"
                print(f"  {config_str:<25} "
                      f"{np.mean(cfg_vals):.4f}±{np.std(cfg_vals):.3f} "
                      f"{np.mean(cfg_tests):.4f}±{np.std(cfg_tests):.3f} "
                      f"{np.mean(cfg_lasts):.4f}±{np.std(cfg_lasts):.3f}")
    
    results['aggregated'] = aggregate_multi_sample_results(results, mlp_param_grid)
    
    total_time = time.time() - total_start
    results['total_time_seconds'] = total_time
    
    if verbose:
        print_summary(results, mlp_param_grid)
        print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    return results


def aggregate_multi_sample_results(
    results: Dict[str, Any],
    mlp_param_grid: List[Tuple]
) -> Dict[str, Any]:
    """Aggregate results across folds and sampled encoders."""
    aggregated = {
        'by_mlp_config': [],
        'by_ali_correlation': {},
    }
    
    for cfg_idx, (hidden, lr, wd, dropout, use_sched) in enumerate(mlp_param_grid):
        all_test_accs = []
        all_test_accs_last = []
        all_val_accs = []
        all_val_accs_last = []
        all_ali_scores = []
        
        for fold_data in results['folds']:
            for sample in fold_data['mpnn_samples']:
                ali_score = sample['ali_score']
                mlp_result = sample['mlp_results'][cfg_idx]
                
                all_test_accs.append(mlp_result['test_acc'])
                all_test_accs_last.append(mlp_result['test_acc_last'])
                all_val_accs.append(mlp_result['val_acc'])
                all_val_accs_last.append(mlp_result['val_acc_last'])
                all_ali_scores.append(ali_score)
        
        n_total = len(all_test_accs)
        
        config_agg = {
            'config_idx': cfg_idx,
            'hidden_layers': hidden,
            'lr': lr,
            'weight_decay': wd,
            'dropout_rate': dropout,
            'use_scheduler': use_sched,
            'n_samples': n_total,
            'test_acc_mean': float(np.mean(all_test_accs)),
            'test_acc_std': float(np.std(all_test_accs, ddof=1)) if n_total > 1 else 0.0,
            'test_acc_min': float(np.min(all_test_accs)),
            'test_acc_max': float(np.max(all_test_accs)),
            'test_acc_last_mean': float(np.mean(all_test_accs_last)),
            'test_acc_last_std': float(np.std(all_test_accs_last, ddof=1)) if n_total > 1 else 0.0,
            'val_acc_mean': float(np.mean(all_val_accs)),
            'val_acc_std': float(np.std(all_val_accs, ddof=1)) if n_total > 1 else 0.0,
            'val_acc_last_mean': float(np.mean(all_val_accs_last)),
            'val_acc_last_std': float(np.std(all_val_accs_last, ddof=1)) if n_total > 1 else 0.0,
            'all_test_accs': all_test_accs,
            'all_val_accs': all_val_accs,
            'all_val_accs_last': all_val_accs_last,
            'all_ali_scores': all_ali_scores,
        }

        aggregated['by_mlp_config'].append(config_agg)

    aggregated['by_mlp_config'].sort(key=lambda x: x['val_acc_mean'], reverse=True)

    best = aggregated['by_mlp_config'][0]
    aggregated['best_config'] = {
        'hidden_layers': best['hidden_layers'],
        'lr': best['lr'],
        'weight_decay': best['weight_decay'],
        'dropout_rate': best['dropout_rate'],
        'val_acc_mean': best['val_acc_mean'],
        'val_acc_std': best['val_acc_std'],
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_std': best['test_acc_std'],
        'selection_metric': 'val_acc_mean',
    }

    best_last = max(aggregated['by_mlp_config'], key=lambda x: x['val_acc_last_mean'])
    aggregated['best_config_last'] = {
        'hidden_layers': best_last['hidden_layers'],
        'lr': best_last['lr'],
        'weight_decay': best_last['weight_decay'],
        'dropout_rate': best_last['dropout_rate'],
        'val_acc_last_mean': best_last['val_acc_last_mean'],
        'val_acc_last_std': best_last['val_acc_last_std'],
        'test_acc_last_mean': best_last['test_acc_last_mean'],
        'test_acc_last_std': best_last['test_acc_last_std'],
        'selection_metric': 'val_acc_last_mean',
    }
    
    all_ali = [s['ali_score'] for s in results['all_ali_scores']]
    aggregated['ali_statistics'] = {
        'mean': float(np.mean(all_ali)),
        'std': float(np.std(all_ali)),
        'min': float(np.min(all_ali)),
        'max': float(np.max(all_ali)),
        'n_samples': len(all_ali),
    }
    
    return aggregated


def print_summary(results: Dict[str, Any], mlp_param_grid: List[Tuple]):
    """Print experiment summary."""
    aggregated = results['aggregated']
    
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (across all MPNN samples and folds)")
    print("="*70)
    
    ali_stats = aggregated['ali_statistics']
    print(f"\nALI Statistics ({ali_stats['n_samples']} samples):")
    print(f"  Mean: {ali_stats['mean']:.4f} ± {ali_stats['std']:.4f}")
    print(f"  Range: [{ali_stats['min']:.4f}, {ali_stats['max']:.4f}]")
    
    print(f"\n--- Top 5 MLP Configurations (by mean VAL acc) ---")
    for i, agg in enumerate(aggregated['by_mlp_config'][:5]):
        print(f"  {i+1}. h={str(agg['hidden_layers']):<12} lr={agg['lr']:.4f} wd={agg['weight_decay']:.0e} "
              f"d={agg['dropout_rate']} | Val: {agg['val_acc_mean']:.4f}±{agg['val_acc_std']:.4f} | "
              f"Test: {agg['test_acc_mean']:.4f}±{agg['test_acc_std']:.4f}")

    by_last = sorted(aggregated['by_mlp_config'], key=lambda x: x['val_acc_last_mean'], reverse=True)
    print(f"\n--- Top 5 MLP Configurations (by mean VAL_LAST acc) ---")
    for i, agg in enumerate(by_last[:5]):
        print(f"  {i+1}. h={str(agg['hidden_layers']):<12} lr={agg['lr']:.4f} wd={agg['weight_decay']:.0e} "
              f"d={agg['dropout_rate']} | ValLast: {agg['val_acc_last_mean']:.4f}±{agg['val_acc_last_std']:.4f} | "
              f"TestLast: {agg['test_acc_last_mean']:.4f}±{agg['test_acc_last_std']:.4f}")

    best = aggregated['best_config']
    best_last = aggregated['best_config_last']
    print(f"\n{'='*70}")
    print(f"BEST CONFIG (by Val): {best['hidden_layers']}, lr={best['lr']}, wd={best['weight_decay']}")
    print(f"  Val Accuracy:  {best['val_acc_mean']:.4f} ± {best['val_acc_std']:.4f}")
    print(f"  Test Accuracy: {best['test_acc_mean']:.4f} ± {best['test_acc_std']:.4f}")
    print(f"\nBEST CONFIG (by ValLast): {best_last['hidden_layers']}, lr={best_last['lr']}, wd={best_last['weight_decay']}")
    print(f"  ValLast Accuracy:  {best_last['val_acc_last_mean']:.4f} ± {best_last['val_acc_last_std']:.4f}")
    print(f"  TestLast Accuracy: {best_last['test_acc_last_mean']:.4f} ± {best_last['test_acc_last_std']:.4f}")
    print("="*70)


def save_results(results: Dict[str, Any], config: MultiSampleConfig) -> str:
    """Save results to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.experiment_name:
        filename = f"{config.experiment_name}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_sample_{config.dataset_name}_{timestamp}.json"
    
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

if __name__ == "__main__":
    config = MultiSampleConfig(
        dataset_name='Mutagenicity',
        n_splits=5,
        cv_seed=0,
        n_mpnn_samples=10,
        sampling=SamplingConfig(
            hidden_channels_list=[256, 256],
            layer_type='GIN',
            pooling='sum',
            weight_init_std=1.0,
        ),
        mlp_hidden_layers=[[64], [64, 64]],
        mlp_lrs=[0.001, 0.0005],
        mlp_weight_decays=[0, 1e-4],
        mlp_dropout_rates=[0.0],
        mlp_use_scheduler=[True],
        mlp_num_epochs=300,
    )
    
    results = run_multi_sample_experiment(config, verbose=True)
    save_path = save_results(results, config)
    print(f"\nResults saved to: {save_path}")
