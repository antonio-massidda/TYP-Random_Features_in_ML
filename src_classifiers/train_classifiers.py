"Training classifiers."

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbor import KNeighborClassifier
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

    train_features = features[train_idx]
    train_labels = labels[train_idx]

    #TODO: Create functions for:

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(Xtrain, y_train)

    return TrainResult(
        val_acc=best_val_acc,
        test_acc=test_acc,
        val_acc_last=best_val_acc_last,
        test_acc_last=test_acc_last,
        model_state=best_model_state
    )
# Logistic Regression
# Gradient Boosting Classifier
# Support Vector Machine
# K-neighbours Classifier