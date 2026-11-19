from typing import Dict, Any, List
from itertools import product
from dataclasses import dataclass
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def get_clfs_config_from_yaml(config: Dict[str, Any], classifier: str = 'all') -> Dict[str, Any]:
    clfs_cfg = config['classifiers']

def expand_param_grid(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """{"a": [1, 2], "b": [3]} -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}]"""
    if not grid:
        return [{}]

    params = list(grid)

    return [dict(zip(params, v)) for v in product(*(grid[k] for k in params))]

@dataclass
class Fold:
    X:      np.array
    y:      np.array

    train:  np.array
    val:    np.array
    test:   np.array


CLASSIFIERS: Dict[str, Any ] = {}
def register_sklearn(name: str, sk_function, scale: bool, seeded: bool):
    """Register a new classifier"""
    class Classifier:
        def __init__(self, fixed=None):
            # self.fixed = fixed
            self.fixed={}

        def fit_eval(self, fold: Fold, params: Dict[str, Any], seed: int):
            args = {**self.fixed, **params} 

            if seeded:
                args["random_state"] = seed

            sk_pipeline = make_pipeline(StandardScaler(), sk_function) if scale else sk_function

            start_time = time.time()
            sk_pipeline.fit(fold.X[fold.train], fold.y[fold.train])
            fit_time = time.time() - start_time
            val_acc = float(sk_pipeline.score(fold.X[fold.val], fold.y[fold.val]))
            test_acc = float(sk_pipeline.score(fold.X[fold.test], fold.y[fold.test]))

            return {
                "val_acc": val_acc,
                "test_acc": test_acc,
                "fit_time": fit_time
            }

    CLASSIFIERS[name] = Classifier

register_sklearn("rf",      RandomForestClassifier, False,  True)
register_sklearn("svm",     SVC,                    True,   False)
register_sklearn("logr",    LogisticRegression,     True,   False)
register_sklearn("knn",     KNeighborsClassifier,   True,   False)


def run_classifiers_on_fold(fold, fold_idx, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """specs: [{'name': 'rf', 'params_list': [...], 'n_seeds': 3, 'fixed': {...}}]"""
    records=[]
    for spec in specs:
        clsf = CLASSIFIERS[spec["name"]]
        for param_config_idx, params in enumerate(spec["params_list"]):
            for seed_idx, seed in enumerate(spec.get("seed"), 1):
                results = clsf.fit_eval(fold, params, seed)
                records += [{
                    "fold": fold_idx, "classifier": clsf,
                    "params_idx": param_config_idx, "params": params,
                    "seed": seed_idx, **results
                }]

    return records


# TODO: aggregator

# TODO: select classifier

# TODO: printing functions