# Notes about original file

---

## What the paper does (in my words)
Sample 2000 random GNNs. Score each with ALI. Keep the best one.
Freeze it. Train a small classifier on its output. No backprop on the GNN.

## File map:
 - `splits.py` -> splits a dataset using the 5-fold cross validation
 - ...

## Questions I have
- [ ] How long does it take to run the program with run.py? 
 - ...

## Things I understand now
 - ...

## Decisions for my project
 - I will keep: mpnn.py, metrics.py, dataset.py, splits.py untouched
 - I will create: train_classical.py, run_classical.py
 - I will modify: sampling.py (add classical branch), config_loader.py

## Experiments I want to run
 1. ...