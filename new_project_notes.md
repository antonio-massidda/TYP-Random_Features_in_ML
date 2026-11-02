# Notes on next work

## What I will do
1. Keep:
   - TU Datasets (or MAYBE graphbench.io)
   - $ALI_k$,
   - 5-fold cross-validation,
   - randomly initialised frozen encoders
   - MLP-on-top approach (as a baseline)
2. Substitute the MLP with four classifiers: RF, LR, SVM, KNN (and MAYBE GBC)
3. New baseline: classifiers on top of one randomly initialised frozen encoder

## New pipeline - dependencies
   ```markdown
   run.py
   ├── config_loader.py
   └── sampling.py
       ├── mpnn.py
       ├── metrics.py
       ├── dataset.py
       |   └── splits.py
       └── train_classifiers.py
   ```

## Questions I have
 - [] Should I use the TU Datasets? Or the graphbench.io? Or both?

## Things I understand now
 - **[X] How long does it take to run the program with run.py?**
    With Teaching HPC and smaller datasets, I can make it work in around half an hour
 - ...

## Implementation decisions for my project
 - I will create: train_classifiers.py
 - I will modify: sampling.py, config_loader.py, run.py, config.yaml
 - I will keep: mpnn.py, metrics.py, dataset.py, splits.py untouched
 - I will baseline: train_mlp.py, mlp.py, utils.py

## Experiments I want to run
 1. Random forest
 2. Logistic regression
 3. Support Vector Machine
 4. $K$-nearest neighbours classifier