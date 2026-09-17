"""End-to-end baseline experiments."""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import copy
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from itertools import product
from tqdm import tqdm

from torch_geometric.utils import spmm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from dataset import GraphDataset, DataSplit, PreparedData
from train_mlp import get_cosine_schedule_with_warmup


class TrainableGCNLayer(nn.Module):
    """Single GCN layer with trainable weights."""
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = spmm(adj, x, reduce='mean')
        return x


class TrainableGINLayer(nn.Module):
    """Single GIN layer with trainable weights."""
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.lin1 = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.lin2 = nn.Linear(out_dim, out_dim, bias=use_bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        aggr = spmm(adj, x, reduce='add')
        h = (1 + self.eps) * x + aggr
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        return h


class TrainableGNN(nn.Module):
    """Full trainable GNN encoder."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels_list: List[int],
        layer_type: str = 'GIN',
        pooling: str = 'sum',
        use_bias: bool = True,
        use_last_activation: bool = True
    ):
        super().__init__()
        self.layer_type = layer_type
        self.pooling = pooling
        self.use_last_activation = use_last_activation
        self.layers = nn.ModuleList()
        
        in_dim = in_channels
        for hidden_dim in hidden_channels_list:
            if layer_type == 'GCN':
                self.layers.append(TrainableGCNLayer(in_dim, hidden_dim, use_bias))
            elif layer_type == 'GIN':
                self.layers.append(TrainableGINLayer(in_dim, hidden_dim, use_bias))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            in_dim = hidden_dim
        
        self.out_dim = hidden_channels_list[-1]
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform (recommended for training)."""
        for layer in self.layers:
            if self.layer_type == 'GCN':
                nn.init.xavier_uniform_(layer.linear.weight)
                if layer.linear.bias is not None:
                    nn.init.zeros_(layer.linear.bias)
            elif self.layer_type == 'GIN':
                nn.init.xavier_uniform_(layer.lin1.weight)
                nn.init.xavier_uniform_(layer.lin2.weight)
                if layer.lin1.bias is not None:
                    nn.init.zeros_(layer.lin1.bias)
                if layer.lin2.bias is not None:
                    nn.init.zeros_(layer.lin2.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1 or self.use_last_activation:
                x = F.relu(x)
        
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            return global_add_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)


class GNNWithMLP(nn.Module):
    """Full end-to-end model: GNN encoder + MLP classifier."""
    def __init__(
        self,
        in_channels: int,
        gnn_hidden_channels: List[int],
        mlp_hidden_layers: List[int],
        num_classes: int,
        layer_type: str = 'GIN',
        pooling: str = 'sum',
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.gnn = TrainableGNN(
            in_channels=in_channels,
            hidden_channels_list=gnn_hidden_channels,
            layer_type=layer_type,
            pooling=pooling
        )
        
        layers = []
        in_dim = gnn_hidden_channels[-1]
        for hidden_dim in mlp_hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor, adj: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Encode graphs into pooled graph embeddings."""
        return self.gnn(x, adj, batch)

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Classify graph embeddings with the MLP head."""
        return self.mlp(embeddings)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode(x, adj, batch)
        return self.classify(embeddings)


@dataclass
class BaselineConfig:
    """Configuration for baseline end-to-end training."""
    dataset_name: str = 'Mutagenicity'
    n_splits: int = 5
    cv_seed: int = 0
    
    device: str = 'cuda:0'
    
    hidden_channels_list: List[int] = field(default_factory=lambda: [256, 256])
    layer_type: str = 'GIN'
    pooling: str = 'sum'
    
    mlp_hidden_layers: List[List[int]] = field(default_factory=lambda: [[64], [64, 64]])
    mlp_dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1])
    
    lrs: List[float] = field(default_factory=lambda: [0.0005, 0.001])
    weight_decays: List[float] = field(default_factory=lambda: [0.0, 1e-4])
    use_scheduler: List[bool] = field(default_factory=lambda: [True])
    num_epochs: int = 300
    warmup_ratio: float = 0.1
    last_n_epochs: int = 20
    n_runs: int = 3
    
    output_dir: str = 'results'
    experiment_name: str = ''

def train_baseline_single(
    data: PreparedData,
    split: DataSplit,
    gnn_hidden: List[int],
    mlp_hidden: List[int],
    layer_type: str,
    pooling: str,
    lr: float,
    weight_decay: float,
    dropout_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    use_scheduler: bool,
    last_n_epochs: int = 20,
    device: torch.device = None
) -> Dict[str, Any]:
    """Train a single end-to-end model."""
    
    if device is None:
        device = data.device
    
    model = GNNWithMLP(
        in_channels=data.num_features,
        gnn_hidden_channels=gnn_hidden,
        mlp_hidden_layers=mlp_hidden,
        num_classes=data.num_classes,
        layer_type=layer_type,
        pooling=pooling,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if use_scheduler:
        num_warmup_steps = int(num_epochs * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_epochs
        )
    else:
        scheduler = None
    
    train_idx = split.train_idx
    val_idx = split.val_idx
    test_idx = split.test_idx
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    best_val_acc_last = 0.0
    best_model_state_last = None
    best_epoch_last = 0
    last_epoch_start = num_epochs - last_n_epochs
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model.encode(data.x, data.adj_csr, data.batch_idx)
        train_out = model.classify(embeddings[train_idx])
        loss = F.cross_entropy(train_out, data.labels[train_idx])
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.adj_csr, data.batch_idx)
                val_pred = out[val_idx].argmax(dim=1)
                val_acc = (val_pred == data.labels[val_idx]).float().mean().item()
            
            if best_model_state is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            
            if epoch >= last_epoch_start:
                if best_model_state_last is None or val_acc > best_val_acc_last:
                    best_val_acc_last = val_acc
                    best_model_state_last = copy.deepcopy(model.state_dict())
                    best_epoch_last = epoch
    
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.adj_csr, data.batch_idx)
        test_pred = out[test_idx].argmax(dim=1)
        test_acc = (test_pred == data.labels[test_idx]).float().mean().item()
    
    if best_model_state_last is not None:
        model.load_state_dict(best_model_state_last)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.adj_csr, data.batch_idx)
            test_pred_last = out[test_idx].argmax(dim=1)
            test_acc_last = (test_pred_last == data.labels[test_idx]).float().mean().item()
    else:
        test_acc_last = test_acc
        best_val_acc_last = best_val_acc
        best_epoch_last = best_epoch
    
    return {
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'val_acc_last': best_val_acc_last,
        'test_acc_last': test_acc_last,
        'best_epoch_last': best_epoch_last
    }


def train_baseline_averaged(
    data: PreparedData,
    split: DataSplit,
    gnn_hidden: List[int],
    mlp_hidden: List[int],
    layer_type: str,
    pooling: str,
    lr: float,
    weight_decay: float,
    dropout_rate: float,
    num_epochs: int,
    warmup_ratio: float,
    use_scheduler: bool,
    n_runs: int,
    last_n_epochs: int = 20,
    device: torch.device = None
) -> Dict[str, Any]:
    """Train multiple times and average results."""
    
    val_accs = []
    test_accs = []
    test_accs_last = []
    val_accs_last = []
    best_epochs = []
    
    for _ in range(n_runs):
        result = train_baseline_single(
            data=data,
            split=split,
            gnn_hidden=gnn_hidden,
            mlp_hidden=mlp_hidden,
            layer_type=layer_type,
            pooling=pooling,
            lr=lr,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            use_scheduler=use_scheduler,
            last_n_epochs=last_n_epochs,
            device=device
        )
        
        val_accs.append(result['val_acc'])
        test_accs.append(result['test_acc'])
        test_accs_last.append(result['test_acc_last'])
        val_accs_last.append(result['val_acc_last'])
        best_epochs.append(result['best_epoch'])
    
    return {
        'val_acc': float(np.mean(val_accs)),
        'val_acc_std': float(np.std(val_accs, ddof=1)) if n_runs > 1 else 0.0,
        'test_acc': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs, ddof=1)) if n_runs > 1 else 0.0,
        'test_acc_last': float(np.mean(test_accs_last)),
        'test_acc_last_std': float(np.std(test_accs_last, ddof=1)) if n_runs > 1 else 0.0,
        'val_acc_last': float(np.mean(val_accs_last)),
        'best_epoch': float(np.mean(best_epochs))
    }

def run_baseline_experiment(
    config: BaselineConfig,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run the baseline experiment with cross-validation."""
    
    device = torch.device(config.device if torch.cuda.is_available() or config.device == 'cpu' else 'cpu')
    
    if verbose:
        print("="*70)
        print("Baseline End-to-End GNN Experiment")
        print("="*70)
        print(f"Dataset: {config.dataset_name}")
        print(f"Device: {device}")
        print(f"CV Folds: {config.n_splits}")
        print(f"GNN: {config.layer_type} {config.hidden_channels_list} pooling={config.pooling}")
        print("="*70, flush=True)
    
    dataset = GraphDataset(name=config.dataset_name, device=device)
    data = dataset.get_prepared_data()
    
    if verbose:
        print(f"Loaded: {data.num_graphs} graphs, {data.num_features} features, {data.num_classes} classes", flush=True)
    
    cv_splits = dataset.get_cross_validation_splits(n_splits=config.n_splits, seed=config.cv_seed)
    
    param_grid = list(product(
        config.mlp_hidden_layers,
        config.lrs,
        config.weight_decays,
        config.mlp_dropout_rates,
        config.use_scheduler
    ))
    
    if verbose:
        print(f"MLP configs: {len(param_grid)}")
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
    
    config_results = {i: [] for i in range(len(param_grid))}
    
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
            'results': []
        }
        
        if verbose:
            print(f"\n--- Training {len(param_grid)} configs (x{config.n_runs} runs each) ---", flush=True)
        
        pbar = tqdm(
            enumerate(param_grid),
            desc="  Training",
            total=len(param_grid),
            disable=not verbose,
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stderr,
            leave=False
        )
        
        for cfg_idx, (mlp_hidden, lr, wd, dropout, use_sched) in pbar:
            result = train_baseline_averaged(
                data=data,
                split=split,
                gnn_hidden=config.hidden_channels_list,
                mlp_hidden=mlp_hidden,
                layer_type=config.layer_type,
                pooling=config.pooling,
                lr=lr,
                weight_decay=wd,
                dropout_rate=dropout,
                num_epochs=config.num_epochs,
                warmup_ratio=config.warmup_ratio,
                use_scheduler=use_sched,
                n_runs=config.n_runs,
                last_n_epochs=config.last_n_epochs,
                device=device
            )
            
            result_dict = {
                'config_idx': cfg_idx,
                'mlp_hidden_layers': mlp_hidden,
                'lr': lr,
                'weight_decay': wd,
                'dropout_rate': dropout,
                'use_scheduler': use_sched,
                **result
            }
            
            fold_result['results'].append(result_dict)
            config_results[cfg_idx].append(result_dict)
            
            pbar.set_postfix({
                'Test': f'{result["test_acc"]:.3f}',
                'Val': f'{result["val_acc"]:.3f}'
            })
        
        pbar.close()
        
        if verbose:
            print(f"\n  {'Config':<25} {'Val':>12} {'Test':>12} {'TestLast':>12}")
            print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
            for res in fold_result['results']:
                h_str = str(res['mlp_hidden_layers'])
                config_str = f"h={h_str:<8} lr={res['lr']:.4f}"
                print(f"  {config_str:<25} "
                      f"{res['val_acc']:.4f}±{res['val_acc_std']:.3f} "
                      f"{res['test_acc']:.4f}±{res['test_acc_std']:.3f} "
                      f"{res['test_acc_last']:.4f}±{res['test_acc_last_std']:.3f}")
        
        results['folds'].append(fold_result)
    
    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("AGGREGATING RESULTS ACROSS FOLDS", flush=True)
        print("="*70, flush=True)
    
    aggregated = []
    for cfg_idx, (mlp_hidden, lr, wd, dropout, use_sched) in enumerate(param_grid):
        fold_results = config_results[cfg_idx]

        test_accs = [r['test_acc'] for r in fold_results]
        test_accs_last = [r['test_acc_last'] for r in fold_results]
        val_accs = [r['val_acc'] for r in fold_results]
        val_accs_last = [r['val_acc_last'] for r in fold_results]
        
        agg = {
            'config_idx': cfg_idx,
            'mlp_hidden_layers': mlp_hidden,
            'lr': lr,
            'weight_decay': wd,
            'dropout_rate': dropout,
            'use_scheduler': use_sched,
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
        'mlp_hidden_layers': best['mlp_hidden_layers'],
        'lr': best['lr'],
        'weight_decay': best['weight_decay'],
        'dropout_rate': best['dropout_rate'],
        'use_scheduler': best['use_scheduler'],
        'val_acc_mean': best['val_acc_mean'],
        'val_acc_std': best['val_acc_std'],
        'test_acc_mean': best['test_acc_mean'],
        'test_acc_std': best['test_acc_std'],
        'selection_metric': 'val_acc_mean'
    }

    aggregated_by_last = sorted(aggregated, key=lambda x: x['val_acc_last_mean'], reverse=True)
    best_last = aggregated_by_last[0]
    results['best_config_last'] = {
        'mlp_hidden_layers': best_last['mlp_hidden_layers'],
        'lr': best_last['lr'],
        'weight_decay': best_last['weight_decay'],
        'dropout_rate': best_last['dropout_rate'],
        'use_scheduler': best_last['use_scheduler'],
        'val_acc_last_mean': best_last['val_acc_last_mean'],
        'val_acc_last_std': best_last['val_acc_last_std'],
        'test_acc_last_mean': best_last['test_acc_last_mean'],
        'test_acc_last_std': best_last['test_acc_last_std'],
        'selection_metric': 'val_acc_last_mean'
    }
    
    total_time = time.time() - total_start
    results['total_time_seconds'] = total_time
    
    if verbose:
        print(f"\n--- Top 5 Configurations (by mean VAL acc across folds) ---")
        for i, agg in enumerate(aggregated[:5]):
            print(f"  {i+1}. h={str(agg['mlp_hidden_layers']):<12} lr={agg['lr']:.4f} wd={agg['weight_decay']:.0e} "
                  f"d={agg['dropout_rate']} | Val: {agg['val_acc_mean']:.4f}±{agg['val_acc_std']:.4f} | "
                  f"Test: {agg['test_acc_mean']:.4f}±{agg['test_acc_std']:.4f}")

        print(f"\n--- Top 5 Configurations (by mean VAL_LAST acc across folds) ---")
        for i, agg in enumerate(aggregated_by_last[:5]):
            print(f"  {i+1}. h={str(agg['mlp_hidden_layers']):<12} lr={agg['lr']:.4f} wd={agg['weight_decay']:.0e} "
                  f"d={agg['dropout_rate']} | ValLast: {agg['val_acc_last_mean']:.4f}±{agg['val_acc_last_std']:.4f} | "
                  f"TestLast: {agg['test_acc_last_mean']:.4f}±{agg['test_acc_last_std']:.4f}")

        print(f"\n{'='*70}")
        print(f"BEST CONFIG (by Val): {best['mlp_hidden_layers']}, lr={best['lr']}, wd={best['weight_decay']}")
        print(f"  Val Accuracy:  {best['val_acc_mean']:.4f} ± {best['val_acc_std']:.4f}")
        print(f"  Test Accuracy: {best['test_acc_mean']:.4f} ± {best['test_acc_std']:.4f}")
        print(f"\nBEST CONFIG (by ValLast): {best_last['mlp_hidden_layers']}, lr={best_last['lr']}, wd={best_last['weight_decay']}")
        print(f"  ValLast Accuracy:  {best_last['val_acc_last_mean']:.4f} ± {best_last['val_acc_last_std']:.4f}")
        print(f"  TestLast Accuracy: {best_last['test_acc_last_mean']:.4f} ± {best_last['test_acc_last_std']:.4f}")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} minutes")
    
    return results


def save_baseline_results(results: Dict[str, Any], config: BaselineConfig) -> str:
    """Save baseline results to JSON file."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.experiment_name:
        filename = f"{config.experiment_name}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baseline_{config.dataset_name}_{timestamp}.json"
    
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
    config = BaselineConfig(
        dataset_name='Mutagenicity',
        n_splits=2,
        hidden_channels_list=[256, 256],
        layer_type='GIN',
        pooling='sum',
        mlp_hidden_layers=[[64]],
        lrs=[0.001],
        weight_decays=[0.0],
        mlp_dropout_rates=[0.0],
        num_epochs=50,
        n_runs=2
    )
    
    results = run_baseline_experiment(config, verbose=True)
    filepath = save_baseline_results(results, config)
    print(f"\nResults saved to: {filepath}")
