"""MLP training helpers."""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from mlp import MLP, MLPConfig
from utils import get_cosine_schedule_with_warmup


@dataclass
class TrainConfig:
    """MLP training configuration."""
    hidden_layers: List[int] = field(default_factory=lambda: [64])
    dropout_rate: float = 0.0
    use_batchnorm: bool = True
    activation: str = 'ReLU'
    
    lr: float = 0.001
    weight_decay: float = 0.0
    num_epochs: int = 300
    
    use_scheduler: bool = True
    warmup_ratio: float = 0.1
    
    last_n_epochs: int = 20


@dataclass
class TrainResult:
    """Results from a single training run."""
    val_acc: float
    test_acc: float
    best_epoch: int
    
    val_acc_last: float
    test_acc_last: float
    best_epoch_last: int
    
    model_state: Optional[Dict[str, Any]] = None


@dataclass
class AveragedTrainResult:
    """Results averaged over multiple training runs."""
    val_acc: float
    val_acc_std: float
    test_acc: float
    test_acc_std: float
    best_epoch: float
    
    val_acc_last: float
    val_acc_last_std: float
    test_acc_last: float
    test_acc_last_std: float
    best_epoch_last: float
    
    model_state: Optional[Dict[str, Any]] = None


def train_mlp(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    config: TrainConfig,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> TrainResult:
    """Train one MLP and keep the best checkpoints by validation accuracy."""
    if device is None:
        device = features.device
    
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    input_dim = features.shape[1]
    num_classes = labels.max().item() + 1
    
    mlp_config = MLPConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=config.hidden_layers,
        dropout_rate=config.dropout_rate,
        use_batchnorm=config.use_batchnorm,
        activation=config.activation
    )
    mlp = MLP(mlp_config).to(device)
    
    optimizer = torch.optim.AdamW(
        mlp.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    if config.use_scheduler:
        num_warmup_steps = int(config.num_epochs * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=config.num_epochs
        )
    else:
        scheduler = None
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    best_val_acc_last = 0.0
    best_model_state_last = None
    best_epoch_last = 0
    
    last_epoch_start = config.num_epochs - config.last_n_epochs
    
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    
    for epoch in range(config.num_epochs):
        mlp.train()
        optimizer.zero_grad()
        out = mlp(train_features)
        loss = F.cross_entropy(out, train_labels)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        if epoch % 5 == 0 or epoch == config.num_epochs - 1:
            mlp.eval()
            with torch.no_grad():
                val_pred = mlp(features[val_idx]).argmax(dim=1)
                val_acc = (val_pred == labels[val_idx]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(mlp.state_dict())
                best_epoch = epoch
            
            if epoch >= last_epoch_start:
                if val_acc > best_val_acc_last:
                    best_val_acc_last = val_acc
                    best_model_state_last = copy.deepcopy(mlp.state_dict())
                    best_epoch_last = epoch
            
            if verbose and (epoch % 10 == 0 or epoch == config.num_epochs - 1):
                current_lr = optimizer.param_groups[0]['lr']
                in_last_phase = "✓" if epoch >= last_epoch_start else " "
                print(f"Ep {epoch:3d} | Loss {loss.item():.4f} | Val {val_acc:.4f} | "
                      f"LR {current_lr:.6f} | Best: ep{best_epoch} {best_val_acc:.4f} {in_last_phase}")
    
    mlp.load_state_dict(best_model_state)
    mlp.eval()
    with torch.no_grad():
        test_pred = mlp(features[test_idx]).argmax(dim=1)
        test_acc = (test_pred == labels[test_idx]).float().mean().item()
    
    if best_model_state_last is not None:
        mlp.load_state_dict(best_model_state_last)
        mlp.eval()
        with torch.no_grad():
            test_pred_last = mlp(features[test_idx]).argmax(dim=1)
            test_acc_last = (test_pred_last == labels[test_idx]).float().mean().item()
    else:
        test_acc_last = test_acc
        best_epoch_last = best_epoch
        best_val_acc_last = best_val_acc
    
    return TrainResult(
        val_acc=best_val_acc,
        test_acc=test_acc,
        best_epoch=best_epoch,
        val_acc_last=best_val_acc_last,
        test_acc_last=test_acc_last,
        best_epoch_last=best_epoch_last,
        model_state=best_model_state
    )


def train_mlp_averaged(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    config: TrainConfig,
    n_runs: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> AveragedTrainResult:
    """Train the same MLP configuration several times and average the results."""
    results: List[TrainResult] = []
    best_test_acc = 0.0
    best_state = None
    
    for run in range(n_runs):
        result = train_mlp(
            features=features,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            config=config,
            device=device,
            verbose=False
        )
        results.append(result)
        
        if result.test_acc > best_test_acc:
            best_test_acc = result.test_acc
            best_state = result.model_state
        
        if verbose:
            print(f"Run {run+1:2d}/{n_runs}: Test={result.test_acc:.4f} "
                  f"TestLast={result.test_acc_last:.4f} Val={result.val_acc:.4f}")
    
    val_accs = [r.val_acc for r in results]
    test_accs = [r.test_acc for r in results]
    best_epochs = [r.best_epoch for r in results]
    val_accs_last = [r.val_acc_last for r in results]
    test_accs_last = [r.test_acc_last for r in results]
    best_epochs_last = [r.best_epoch_last for r in results]
    
    return AveragedTrainResult(
        val_acc=float(np.mean(val_accs)),
        val_acc_std=float(np.std(val_accs, ddof=1)),
        test_acc=float(np.mean(test_accs)),
        test_acc_std=float(np.std(test_accs, ddof=1)),
        best_epoch=float(np.mean(best_epochs)),
        val_acc_last=float(np.mean(val_accs_last)),
        val_acc_last_std=float(np.std(val_accs_last, ddof=1)),
        test_acc_last=float(np.mean(test_accs_last)),
        test_acc_last_std=float(np.std(test_accs_last, ddof=1)),
        best_epoch_last=float(np.mean(best_epochs_last)),
        model_state=best_state
    )


if __name__ == "__main__":
    from dataset import GraphDataset
    
    print("Loading dataset...")
    dataset = GraphDataset(name='Mutagenicity')
    data = dataset.get_prepared_data()
    split = dataset.get_single_split(seed=0)
    
    print(f"Dataset: {data.num_graphs} graphs, {data.num_features} features")
    print(f"Split: train={len(split.train_idx)}, val={len(split.val_idx)}, test={len(split.test_idx)}")
    
    print("\nCreating dummy features for testing...")
    torch.manual_seed(42)
    dummy_features = torch.randn(data.num_graphs, 128, device=data.device)
    
    print("\n" + "="*50)
    print("Testing single training run...")
    print("="*50)
    
    config = TrainConfig(
        hidden_layers=[64],
        lr=0.001,
        num_epochs=100,
        dropout_rate=0.0,
    )
    
    result = train_mlp(
        features=dummy_features,
        labels=data.labels,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
        config=config,
        verbose=True
    )
    
    print(f"\nResult:")
    print(f"  Val Acc: {result.val_acc:.4f} (best epoch: {result.best_epoch})")
    print(f"  Test Acc: {result.test_acc:.4f}")
    print(f"  Test Acc (last {config.last_n_epochs}): {result.test_acc_last:.4f}")
    
    print("\n" + "="*50)
    print("Testing averaged training (5 runs)...")
    print("="*50)
    
    avg_result = train_mlp_averaged(
        features=dummy_features,
        labels=data.labels,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
        config=config,
        n_runs=5,
        verbose=True
    )
    
    print(f"\nAveraged Result:")
    print(f"  Val Acc: {avg_result.val_acc:.4f} ± {avg_result.val_acc_std:.4f}")
    print(f"  Test Acc: {avg_result.test_acc:.4f} ± {avg_result.test_acc_std:.4f}")
    print(f"  Test Acc (last {config.last_n_epochs}): {avg_result.test_acc_last:.4f} ± {avg_result.test_acc_last_std:.4f}")
