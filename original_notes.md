# Notes about original file

---

## What the paper does (in my words)
Sample 2000 random GNNs. Score each with ALI. Keep the best one.
Freeze it. Train a small classifier on its output. No backprop on the GNN.

## File map
 1. `splits.py` -> splits a dataset using the 5-fold cross validation
 2. `dataset.py` -> gets dataset information (e.g. name and root) and applies the split
 3. `metrics.py` -> computes ALI
 4. `mlp.py` -> applies MLP using steps `GNN layer` -> `BatchNorm` -> `ReLU` -> `Next Layer`
 5. `utils.py` -> handles learning rates by increasing them at a warm-up stage and gradually decreasing them later using the cosine function
 6. `train_mlp` -> trains the model using MLP and releases stats about it
 7. `mpnn.py` -> describe classes of Message Passing Neural Networks
 8. `sampling.py` -> find best of 2000 mpnn
 9. `sampling_multi.py` -> sampling.py but repeated 10 times 
 10. `baseline.py` -> usual GNN end-to-end approach
 11. `config_loader.py` -> loads the YAML file to use
 12. `run.py` -> runs the experiment completely
 13. `run_multi.py` -> runs the experiment 10 times
 14. `run_baseline.py` -> runs the usual GNN end-to-end approach

## Dependencies
   ```
   run.py
   ├── config_loader.py
   └── sampling.py
       ├── mpnn.py
       ├── metrics.py
       ├── dataset.py
       |   └── splits.py
       └── train_mlp.py
           ├── mlp.py
           └── utils.py
   ```
       

## Questions I have
 - ...

## Things I understand now
 - **[X] How long does it take to run the program with run.py?**
    With Teaching HPC and smaller datasets, I can make it work in around half a hour
 - ...

## Decisions for my project
 - I will keep: mpnn.py, metrics.py, dataset.py, splits.py untouched
 - I will delete: train_mlp.py, mlp.py, utils.py
 - I will create: train_classical, classical.py
 - I will modify: sampling.py, config_loader.py, run.py

## Experiments I want to run
 1. Random forest
 3. Logistic regression
 4. Gradient boosting classifier
 5. Support Vector Machine
 6. K-neighbours classifier