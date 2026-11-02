# Notes about original file

## What the paper does (in my words)
1. Load 7 benchmark TU Datasets
1. Apply a 5-fold cross-validation ($80\%, 10\%, 10\%$) split protocol
1. Sample 2,000 randomly initialised GNN frozen encoders
1. Calculate an alignment score ($ALI_k$) for all encoders.  
   - Based on a KNN, where similarity with $k$-nearest is subtracted to similarity with all the others.  
   - The comparison is only within the training set.
1. Select the frozen encoder with the highest $ALI_k$
   - Train an MLP on top of it (16 different configurations can be used regarding hidden layers, GIN-GCN, and mean-sum poolin)
1. Repeat this experiments for all folds and for all seed (5 x 3 = 15 runs in total)
1. Compare results to its end-to-end MLP baseline between models and for each dataset
1. Vargha-Delaney $A_{12}$ measurement = 0.48 :arrow_right: NO SIGNIFICANT DIFFERENCE


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