"""Dataset utilities."""

import warnings
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_torch_coo_tensor, degree
from torch_geometric.transforms import BaseTransform
from dataclasses import dataclass
from typing import Optional, List, Tuple

from splits import generate_stratified_80_10_10_splits

# Silence PyTorch's CSR beta warning for the adjacency conversion below.
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")

DATASETS_WITHOUT_FEATURES = {
    'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
    'IMDB-BINARY', 'IMDB-MULTI'}


def compute_unique_degrees(dataset: TUDataset) -> List[int]:
    """Return the sorted set of node degrees in the dataset."""
    all_degrees = set()
    for data in dataset:
        row, _ = data.edge_index
        deg = degree(row, data.num_nodes, dtype=torch.long)
        all_degrees.update(deg.tolist())
    return sorted(all_degrees)


class AddDegreeFeatures(BaseTransform):
    """Add one-hot degree features."""
    
    def __init__(self, degree_to_idx: dict, cat: bool = False):
        self.degree_to_idx = degree_to_idx
        self.num_features = len(degree_to_idx)
        self.cat = cat
    
    def forward(self, data: Data) -> Data:
        row, col = data.edge_index
        deg = degree(row, data.num_nodes, dtype=torch.long)
        
        deg_idx = torch.tensor([self.degree_to_idx[d.item()] for d in deg], dtype=torch.long)
        
        deg_onehot = torch.zeros(data.num_nodes, self.num_features)
        deg_onehot.scatter_(1, deg_idx.unsqueeze(1), 1)
        
        if data.x is None or not self.cat:
            data.x = deg_onehot
        else:
            data.x = torch.cat([data.x, deg_onehot], dim=-1)
        
        return data


@dataclass
class DataSplit:
    """Train/val/test split."""
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    fold: int = 0


@dataclass
class PreparedData:
    """Prepared graph tensors."""
    x: torch.Tensor
    adj_csr: torch.Tensor
    batch_idx: torch.Tensor
    labels: torch.Tensor
    num_features: int
    num_classes: int
    num_graphs: int
    device: torch.device
    

class GraphDataset:
    """Load a TUDataset and prepare batched tensors."""
    
    def __init__(
        self, 
        name: str = 'Mutagenicity',
        root: str = '/tmp',
        device: Optional[torch.device] = None,
        use_degree_features: str = 'auto',
    ):
        self.name = name
        self.root = root
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_degree_features = use_degree_features
        
        self._dataset = TUDataset(root=f'{root}/{name}', name=name)
        needs_degree_features = self._check_needs_degree_features()
        
        if needs_degree_features:
            print(f"Dataset '{name}' has no node features. Adding degree features...")
            unique_degrees = compute_unique_degrees(self._dataset)
            degree_to_idx = {d: i for i, d in enumerate(unique_degrees)}
            
            print(f"  Found {len(unique_degrees)} unique degrees (min={unique_degrees[0]}, max={unique_degrees[-1]})")
            
            transform = AddDegreeFeatures(degree_to_idx=degree_to_idx)
            self._dataset = TUDataset(
                root=f'{root}/{name}', 
                name=name,
                transform=transform
            )
            self._num_features = len(unique_degrees)
            print(f"  Added {self._num_features} degree-based features")
        else:
            self._num_features = self._dataset.num_features
        
        self._prepare_data()
    
    def _check_needs_degree_features(self) -> bool:
        """Check if the dataset needs degree features."""
        if self.use_degree_features == False:
            return False
        
        if self.use_degree_features == True:
            return True
        
        if self.name in DATASETS_WITHOUT_FEATURES:
            return True
        
        first_data = self._dataset[0]
        if first_data.x is None:
            return True
        
        return False
    
    def _prepare_data(self):
        """Build the batched tensors used by the models."""
        self._full_batch = Batch.from_data_list([d for d in self._dataset]).to(self.device)
        
        adj = to_torch_coo_tensor(
            self._full_batch.edge_index, 
            size=(self._full_batch.num_nodes, self._full_batch.num_nodes)
        )
        self._adj_csr = adj.float().to_sparse_csr().to(self.device)
        
        self._x = self._full_batch.x.float().to(self.device)
        self._labels = self._full_batch.y.to(self.device)
    
    @property
    def num_graphs(self) -> int:
        return len(self._dataset)
    
    @property
    def num_features(self) -> int:
        return self._num_features
    
    @property
    def num_classes(self) -> int:
        return self._dataset.num_classes
    
    def get_prepared_data(self) -> PreparedData:
        """Return the prepared tensors."""
        return PreparedData(
            x=self._x,
            adj_csr=self._adj_csr,
            batch_idx=self._full_batch.batch,
            labels=self._labels,
            num_features=self.num_features,
            num_classes=self.num_classes,
            num_graphs=self.num_graphs,
            device=self.device
        )
    
    def get_cross_validation_splits(
        self,
        n_splits: int = 5,
        seed: int = 0
    ) -> List[DataSplit]:
        """Generate deterministic 80/10/10 stratified splits."""
        raw_splits = generate_stratified_80_10_10_splits(
            self._labels,
            n_splits=n_splits,
            seed=seed,
        )

        return [
            DataSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, fold=fold)
            for fold, (train_idx, val_idx, test_idx) in enumerate(raw_splits)
        ]
    
    def get_single_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 0
    ) -> DataSplit:
        """Return a single random split."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        torch.manual_seed(seed)
        N = self.num_graphs
        perm = torch.randperm(N)
        
        train_size = int(train_ratio * N)
        val_size = int(val_ratio * N)
        
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]
        test_idx = perm[train_size + val_size:]
        
        return DataSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            fold=0
        )
    
    def get_labels_for_split(self, split: DataSplit) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return labels for a split."""
        return (
            self._labels[split.train_idx],
            self._labels[split.val_idx],
            self._labels[split.test_idx]
        )
    
    def __repr__(self) -> str:
        return (
            f"GraphDataset(name='{self.name}', "
            f"num_graphs={self.num_graphs}, "
            f"num_features={self.num_features}, "
            f"num_classes={self.num_classes}, "
            f"device={self.device})"
        )


if __name__ == '__main__':
    print("=" * 60)
    print("Testing with Mutagenicity (has node features)")
    print("=" * 60)
    dataset = GraphDataset(name='Mutagenicity')
    print(dataset)
    
    data = dataset.get_prepared_data()
    print(f"\nPrepared data (all on {data.device}):")
    print(f"  Features shape: {data.x.shape} on {data.x.device}")
    print(f"  Adj CSR shape: {data.adj_csr.shape} on {data.adj_csr.device}")
    print(f"  Labels shape: {data.labels.shape} on {data.labels.device}")
    print(f"  Batch idx shape: {data.batch_idx.shape} on {data.batch_idx.device}")
    print(f"  Num graphs: {data.num_graphs}")
    
    print(f"\n--- Single Split (80/10/10) ---")
    split = dataset.get_single_split()
    print(f"  Train: {len(split.train_idx)} ({len(split.train_idx)/data.num_graphs*100:.1f}%)")
    print(f"  Val:   {len(split.val_idx)} ({len(split.val_idx)/data.num_graphs*100:.1f}%)")
    print(f"  Test:  {len(split.test_idx)} ({len(split.test_idx)/data.num_graphs*100:.1f}%)")
    
    print(f"\n--- Cross-Validation (5 folds) ---")
    cv_splits = dataset.get_cross_validation_splits(n_splits=5)
    for split in cv_splits:
        print(f"  Fold {split.fold}: Train={len(split.train_idx)}, Val={len(split.val_idx)}, Test={len(split.test_idx)}")
    
    print(f"\n--- Verifying no overlap in fold 0 ---")
    s = cv_splits[0]
    train_set = set(s.train_idx.tolist())
    val_set = set(s.val_idx.tolist())
    test_set = set(s.test_idx.tolist())
    print(f"  Train ∩ Val: {len(train_set & val_set)}")
    print(f"  Train ∩ Test: {len(train_set & test_set)}")
    print(f"  Val ∩ Test: {len(val_set & test_set)}")
    print(f"  Total coverage: {len(train_set | val_set | test_set)} / {data.num_graphs}")
    
    print("\n" + "=" * 60)
    print("Testing with IMDB-BINARY (no node features - uses degree)")
    print("=" * 60)
    try:
        dataset2 = GraphDataset(name='IMDB-BINARY', device=torch.device('cpu'))
        print(dataset2)
        data2 = dataset2.get_prepared_data()
        print(f"\nPrepared data:")
        print(f"  Features shape: {data2.x.shape} (degree-based)")
        print(f"  Num graphs: {data2.num_graphs}")
        print(f"  Num classes: {data2.num_classes}")
    except Exception as e:
        print(f"Could not load IMDB-BINARY: {e}")
