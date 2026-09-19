# Notes about original file

---

## What the paper does (in my words)
Sample 2000 random GNNs. Score each with ALI. Keep the best one.
Freeze it. Train a small classifier on its output. No backprop on the GNN.

## File map:
 1. `splits.py` -> splits a dataset using the 5-fold cross validation
 2. `dataset.py` -> gets dataset information (e.g. name and root) and applies the split
 3. `metrics.py` -> computes ALI
 4. `mlp.py` -> applies MLP using steps `GNN layer` -> `BatchNorm` -> `ReLU` -> `Next Layer`
 5. `utils.py` -> handles learning rates by increasing them at a warm-up stage and gradually decreasing them later using the cosine function
 6. `train_mlp` -> trains the model using MLP and releases stats about it
 12. ...

## Questions I have
 - ...

## Things I understand now
 - **[X] How long does it take to run the program with run.py?**
    With Teaching HPC and smaller datasets, I can make it work in around half a hour
 - ...

## Decisions for my project
 - I will keep: mpnn.py, metrics.py, dataset.py, splits.py untouched
 - I will delete: mlp.py
 - I will create: train_classical.py, run_classical.py
 - I will modify: sampling.py (add classical branch), config_loader.py

## Experiments I want to run
 1. ...