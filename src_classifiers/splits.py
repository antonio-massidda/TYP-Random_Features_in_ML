"""
Utilities for deterministic stratified train/val/test splitting.
"""

from typing import List, Tuple

import torch


def _shuffle_indices(indices: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return a deterministically shuffled copy of indices."""
    if indices.numel() <= 1:
        return indices.clone()
    perm = torch.randperm(indices.numel(), generator=generator)
    return indices[perm]


def generate_stratified_80_10_10_splits(
    labels: torch.Tensor,
    n_splits: int = 5,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build stratified 80/10/10 splits across ``n_splits`` rounds.

    Protocol:
    - Partition each class into ``n_splits`` disjoint outer folds.
    - For fold ``i``, use outer fold ``i`` as the 20% holdout set.
    - Split that holdout set again, per class, into 50/50 val/test halves.
    - The remaining outer folds form the 80% train set.

    This matches a 5-round protocol where each sample is held out exactly once
    across all rounds, while each round uses roughly 80% train, 10% val, 10% test.
    """
    if labels.dim() != 1:
        raise ValueError("labels must be a 1D tensor")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    labels_cpu = labels.detach().cpu()
    generator = torch.Generator().manual_seed(seed)

    per_class_outer_folds = {}
    for label in torch.unique(labels_cpu, sorted=True):
        class_indices = torch.nonzero(labels_cpu == label, as_tuple=True)[0]
        class_indices = _shuffle_indices(class_indices, generator)
        per_class_outer_folds[int(label.item())] = [
            chunk.clone() for chunk in torch.tensor_split(class_indices, n_splits)
        ]

    splits: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for fold in range(n_splits):
        train_parts = []
        val_parts = []
        test_parts = []

        for outer_folds in per_class_outer_folds.values():
            holdout = outer_folds[fold]
            train_chunks = outer_folds[:fold] + outer_folds[fold + 1 :]
            train_parts.append(torch.cat(train_chunks) if train_chunks else holdout.new_empty(0))

            shuffled_holdout = _shuffle_indices(holdout, generator)
            val_chunk, test_chunk = torch.tensor_split(shuffled_holdout, 2)
            val_parts.append(val_chunk.clone())
            test_parts.append(test_chunk.clone())

        train_idx = torch.cat(train_parts).sort().values
        val_idx = torch.cat(val_parts).sort().values
        test_idx = torch.cat(test_parts).sort().values
        splits.append((train_idx, val_idx, test_idx))

    return splits
