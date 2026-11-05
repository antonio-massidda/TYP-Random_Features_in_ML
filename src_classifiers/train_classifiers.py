"Training classifiers."

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass, field
from typing import List

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


def train_classifiers(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    config: TrainConfig,
    device: Optional[torch.device] = None,
    verbose: bool = False
):
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    features = features.to(device)
    labels = labels.to(device)

    num_classes = labels.max().item() + 1

    best_val_acc = 0.0
    best_model_state = None

    best_val_acc_last = 0.0
    best_model_state_last = None

    train_features, train_labels    = features[train_idx],  labels[train_idx]
    val_features,   val_labels      = features[val_idx],    labels[val_idx]
    test_features,  test_labels     = features[test_idx],   labels[test_idx]


    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(train_features, train_labels)

    return TrainResult(
        classifier="rf",
        val_acc=rf.score(val_features, val_labels),
        test_acc=rf.score(test_features, test_labels),
    )


def train_classifiers_averaged(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    config: TrainConfig,
    n_runs: int = 3,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> AveragedTrainResult:
    """Train the same classifiers configuration several times and average the results."""
    results: List[TrainResult] = []
    best_test_acc = 0.0
    
    for run in range(n_runs):
        result = train_classifiers(
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
    
    return AveragedTrainResult(
        val_acc=float(np.mean(val_accs)),
        val_acc_std=float(np.std(val_accs, ddof=1)),
        test_acc=float(np.mean(test_accs)),
        test_acc_std=float(np.std(test_accs, ddof=1)),
    )

#TODO: Create functions for:
# Logistic Regression
# Support Vector Machine
# K-neighbours Classifier