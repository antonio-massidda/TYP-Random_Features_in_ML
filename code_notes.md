# PyTorch / Code Notes

> Short reference notes for understanding the project repository.

---

## `torch.tensor()`

**What it does:**  
Creates a PyTorch tensor, the main data structure used for storing and processing numerical data.

**Examples:**
```python
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]])
```

---

## `numel()`

**What it does:**  
Returns the total number of elements in a tensor.

**Example:**
```python
x = torch.tensor([[1, 2], [3, 4]])
x.numel()  # 4
```

---

## `clone()`

**What it does:**
Creates a copy of a tensor, preserving its values while giving it separate storage.

**Example:**
```python
y = x.clone()
```
**Remember:**
Useful when you want to modify a tensor without modifying the original.

---

## `randperm()`

**What it does:**
Returns a random permutation of integers from 0 to n-1.

**Example:**
```python
torch.randperm(5)
# tensor([3, 0, 4, 1, 2])
```
**Remember:**
Often used to randomly shuffle indices.

---

## `detach()`

**What it does:**
Creates a tensor that shares the same data but is disconnected from the computation graph.

**Example:**
```python
y = x.detach()
```
**Remember:**
Useful when you want to use a tensor's value without calculating gradients through it.

---

## `cpu()`

**What it does:**
Moves a tensor to the CPU.

**Example:**
```python
x = x.cpu()
```
**Remember:**
Often needed before converting a GPU tensor to NumPy:
```python
x.cpu().numpy()
```
---

## `Generator()`

**What it does:**
Creates a PyTorch random-number generator whose state can be controlled independently.

**Example:**
```python
g = torch.Generator()
```
**Remember:**
Useful for reproducible random operations.

---

## `manual_seed()`

**What it does:**
Sets the seed used by a random-number generator, making random operations reproducible.

**Example:**
```python
torch.manual_seed(42)
```
**Remember:**
Same seed + same operation → usually the same random result.

---

## `unique()`

**What it does:**
Returns the unique values contained in a tensor.

**Example:**
```python
x = torch.tensor([1, 2, 2, 3, 3])
torch.unique(x)
# tensor([1, 2, 3])
```
**Remember:**
Useful for finding distinct classes, labels, or values.

---

## `nonzero()`

**What it does:**
Returns the indices of elements that are not zero.

**Example:**
```python
x = torch.tensor([0, 5, 0, 3])
torch.nonzero(x)
# tensor([[1], [3]])
```
**Remember:**
Think: "Where are the non-zero elements?"

---

## `item()`

**What it does:**
Converts a single-element tensor into a standard Python number.

**Example:**
```python
x = torch.tensor(5.0)
x.item()
# 5.0
```
**Remember:**
The tensor must contain exactly one element.

---

## `tensor_split()`

**What it does:**
Splits a tensor into multiple smaller tensors along a specified dimension.

**Example:**
```python
x = torch.arange(6)
torch.tensor_split(x, 3)
# [tensor([0, 1]), tensor([2, 3]), tensor([4, 5])]
```
**Remember:**
Useful for dividing data into subsets/chunks.

---

## `values()`

**What it does:**
Returns the values from a PyTorch sparse tensor or certain tensor-like structures.

**Example:**
```python
values = sparse_tensor.values()
```
**Remember:**
Usually encountered when working with sparse tensors, where values and indices are stored separately.

---

## `new_empty()`

**What it does:**
Creates an uninitialized tensor with the same dtype and device as an existing tensor.

**Example:**
```python
x = torch.tensor([1, 2, 3])
y = x.new_empty(2, 3)
```
**Remember:**
"Create a new empty tensor like this one."

---

## `cat()`

**What it does:**
Concatenates tensors along an existing dimension.

**Example:**
```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])

torch.cat([a, b])
# tensor([1, 2, 3, 4])
```
**Remember:**
The tensors must have compatible shapes except along the dimension being concatenated.

---

## `sort()`

**What it does:**
Sorts tensor elements and returns both the sorted values and their original indices.

**Example:**
```python
x = torch.tensor([3, 1, 2])
values, indices = torch.sort(x)

# values  -> [1, 2, 3]
# indices -> [1, 2, 0]
```
**Remember:**
values = sorted elements; indices = where they came from.


One particularly useful distinction for your repository: **`clone()` vs `detach()`** — `clone()` concerns copying data, while `detach()` concerns **gradient tracking/computation graphs**.

---

## `Batch`

**What it does:**
A PyTorch Geometric object that combines multiple `Data` graphs into a single batch for processing by a GNN.

**Example:**
```python
batch = Batch.from_data_list([graph1, graph2])
```
**Remember:**
Allows multiple graphs to be processed together while keeping track of which nodes belong to each graph.

---

## `Data`

**What it does:**
The main PyTorch Geometric object for representing a graph, containing attributes such as nodes, edges, and labels.

**Example:**
```python
data = Data(x=x, edge_index=edge_index)
```
**Remember:**
data.x usually stores node features, while data.edge_index stores graph connectivity.

---

## `TUDataset`

**What it does:**
Loads graph classification datasets from the TU Dortmund graph database collection.

**Example:**
```python
dataset = TUDataset(root="data/", name="MUTAG")
```
**Remember:**
Commonly used for graph classification experiments and benchmarking GNNs.

---

## `to_torch_coo_tensor()`

**What it does:**
Converts graph connectivity information, such as edge_index, into a PyTorch sparse COO tensor.

**Example:**
```python
adj = to_torch_coo_tensor(data.edge_index)
```
**Remember:**
COO (Coordinate) format represents a sparse matrix using its non-zero entries and their indices.

---

## `degree()`

**What it does:**
Calculates the degree of each node, i.e. the number of edges connected to it.

**Example:**
```python
deg = degree(data.edge_index[0])
```
**Remember:**
In an undirected graph, a node's degree is the number of neighbours it has.

---

## `BaseTransform`

**What it does:**
Base class used to create custom transformations that modify or process PyTorch Geometric Data objects.

**Example:**
```python
class MyTransform(BaseTransform):
    def forward(self, data):
        return data
```
**Remember:**
Transforms are commonly applied automatically when loading or processing graph data.

---

## `data.edge_index`

**What it does:**
Stores the graph's edges as a tensor, usually with shape [2, number_of_edges].

**Example:**
```python
edge_index = data.edge_index
```
**Remember:**
The first row contains source nodes and the second row contains destination nodes.

---

## `data.num_nodes`

**What it does:**
Returns the number of nodes in the graph.

**Example:**
```python
n = data.num_nodes
```
**Remember:**
Useful when creating tensors or iterating over all nodes.

---

## `torch.long`

**What it does:**
Specifies the 64-bit integer (int64) data type used by PyTorch tensors.

**Example:**
```python
x = torch.tensor([1, 2, 3], dtype=torch.long)
```
**Remember:**
Frequently used for indices, labels, and edge_index.

---

## `torch.zeros`

**What it does:**
Creates a tensor filled entirely with zeros.

**Example:**
```python
x = torch.zeros(3, 4)
```
**Remember:**
Useful for initialising tensors before filling them with values.

---

## `scatter_`

**What it does:**
Writes values from one tensor into another tensor at specified indices, modifying the target tensor in place.

**Example:**
```python
x.scatter_(0, index, values)
```
**Remember:**
The _ means the operation modifies the tensor directly rather than creating a new tensor.

---

## `torch.device`

**What it does:**
Represents the device where a tensor or model is stored, typically the CPU or a CUDA GPU.

**Example:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
**Remember:**
Models and tensors involved in the same computation generally need to be on the same device.

---

## `tensor.unsqueeze()`

**What it does:**
Adds a new dimension of size `1` to a tensor at the specified position.

**Example:**
```python
x = torch.tensor([1, 2, 3])
x = x.unsqueeze(0)

# Shape: [3] → [1, 3]
```
**Remember:**
Useful when you need to change a tensor's shape without changing its values, e.g. to add a batch dimension.

---

## `import torch.nn.functional as F`

**What it does:**
Imports PyTorch's functional API, which provides functions for operations such as activation functions, losses, and convolutions.

**Important:**
- `F` → alias used to access the functions.

**Example:**
```python
import torch.nn.functional as F

x = F.relu(x)
loss = F.cross_entropy(output, target)
```
**Remember:**
F.relu() performs an operation without creating a layer/object like nn.ReLU().

---

## `tensor.permute()`

**What it does:**
Rearranges the dimensions of a tensor into a specified order.

**Important arguments:** 

- `*dims` → order in which the existing dimensions should appear.

**Example:**
```python
x = x.permute(0, 2, 1)
```
**Remember:**
Changes the dimension order, not the values.

---

## `cdist()`

**What it does:**
Computes pairwise distances between two sets of vectors.

**Important arguments:** 

- `x1` → first set of vectors.
- `x2` → second set of vectors.
- `p` → type of Minkowski distance; p=2 means Euclidean distance.

**Example:**
```python
dist = torch.cdist(x1, x2, p=2)
```
**Remember:**
cdist() answers: "How far is every vector in x1 from every vector in x2?"

---

## `float()`

**What it does:**
Converts a tensor to the float32 data type.

**Important arguments:** 

- `None` → converts the tensor to the default floating-point type (torch.float32).

**Example:**
```python
x = x.float()
```
**Remember:**
Useful when numerical operations require floating-point values.

---

## `view()`

**What it does:**
Reshapes a tensor without changing its values.

**Important arguments:** 

- `shape` → the desired dimensions.
- `-1` → automatically calculates that dimension based on the other dimensions.

**Example:**
```python
x = x.view(-1, 128)
```
**Remember:**
The total number of elements must remain the same.

---

## `float().view()`

**What it does:**
First converts a tensor to float32, then reshapes it.

**Important arguments:** 

- `float()` → converts the data type to float32.
- `view(shape)` → specifies the new tensor shape.
- `view(-1)` → automatically infers that dimension.

**Example:**
```python
x = x.float().view(-1, 128)
```
**Remember:**
Read it from left to right: convert dtype → reshape.

---

## `arange()`

**What it does:**
Creates a 1D tensor containing evenly spaced values within a specified range.

**Important arguments:** 

- `start` → first value (default 0).
- `end` → stopping boundary; normally not included.
- `step` → difference between consecutive values (default 1).
- `dtype` → optional data type.

**Example:**
```python
torch.arange(5)
# tensor([0, 1, 2, 3, 4])

torch.arange(2, 10, 2)
# tensor([2, 4, 6, 8])
```
**Remember:**
Similar to Python's range(), but returns a PyTorch tensor.

---

## `torch.nn`

**What it does:**
PyTorch's neural-network module containing layers, activation functions, loss functions, and other building blocks for creating models.

**Example:**
```python
import torch.nn as nn

layer = nn.Linear(10, 5)
```
**Remember:**
nn is commonly used to define the structure of a neural network.

---

## `nn.ModuleList()`

**What it does:**
Stores a list of PyTorch modules while ensuring that their parameters are properly registered by the model.

**Important arguments:**

- `modules` → optional iterable of layers/modules to store.

**Example:**
```python
layers = nn.ModuleList([
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
])
```
**Remember:**
Use ModuleList instead of a normal Python list when the layers need to be recognised as part of the model.

---

## `nn.ReLU()`

**What it does:**
Creates a ReLU activation layer: max(0, x). Negative values become 0, while positive values are unchanged.

**Important arguments:**

- `inplace` → if True, modifies the input tensor directly.

**Example:**
```python
relu = nn.ReLU()
x = relu(x)
```
**Remember:**
ReLU introduces non-linearity into a neural network.

---

## `nn.LeakyReLU()`

**What it does:**
Creates a ReLU-like activation that keeps a small negative value instead of setting negative inputs to zero.

**Important arguments:**

- `negative_slope` → multiplier applied to negative input values.
- `inplace` → if True, modifies the input tensor directly.

**Example:**
```python
activation = nn.LeakyReLU(negative_slope=0.01)
x = activation(x)
```
**Remember:**
Unlike ReLU, negative inputs still produce a small gradient.

---

## `nn.Linear()`

**What it does:**
Applies a linear transformation to the input:

y = xWᵀ + b

**Important arguments:**

- `in_features` → number of input features.
- `out_features` → number of output features.
- `bias` → whether to include a learnable bias term.

**Example:**
```python
layer = nn.Linear(128, 64)
x = layer(x)
```
**Remember:**
128 → 64 means 128 input features become 64 output features.

---

## `nn.BatchNorm1d()`

**What it does:**
Normalizes features across a batch to help stabilize and accelerate neural-network training.

**Important arguments:**

- `num_features` → number of features/channels being normalized.
- `eps` → small value added for numerical stability.
- `momentum` → controls how running statistics are updated.
- `affine` → if True, learns scaling and shifting parameters.

**Example:**
```python
bn = nn.BatchNorm1d(64)
x = bn(x)
```
**Remember:**
num_features normally corresponds to the feature dimension of the input.

---

## `F.dropout(x, p=, training=)`

**What it does:**
Randomly sets some elements of x to zero during training to reduce overfitting.

**Important arguments:**

- `x` → input tensor.
- `p` → probability of an element being set to zero.
- `training` → whether dropout is active; normally set to the model's training state.

**Example:**
```python
x = F.dropout(x, p=0.5, training=self.training)
```
**Remember:**
p=0.5 means approximately 50% of elements are dropped during training. Dropout is disabled during evaluation.

---

## `flatten()`

**What it does:**
Flattens consecutive dimensions of a tensor into a single dimension.

**Important arguments:**

- `start_dim` → first dimension to flatten.
- `end_dim` → last dimension to flatten.

**Example:**
```python
x = x.flatten(start_dim=1)

If x has shape [batch, channels, height, width]:

[32, 3, 28, 28] → [32, 2352]
```
**Remember:**
start_dim=1 is commonly used to flatten everything except the batch dimension.

---

## `import torch.optim as optim`

**What it does:**
Imports PyTorch's optimization package, which provides optimizers such as SGD and Adam for updating model parameters.

**Important:**
- `optim` → alias used to access optimizers and learning-rate schedulers.

**Example:**
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Remember:**
The optimizer uses gradients to update the model's weights.

---

## `from torch.optim import Optimizer`

**What it does:**
Imports the base Optimizer class used by PyTorch optimizers.

**Important:**

 - `Optimizer` → base class for implementing and managing parameter optimization.

**Example:**
```python
from torch.optim import Optimizer

optimizer = Optimizer(params, defaults)
```
**Remember:**
Usually encountered when implementing or type-checking custom optimizers, rather than normal model training.

---

## `optim.lr_scheduler.LambdaLR()`

**What it does:**
Creates a learning-rate scheduler that adjusts the optimizer's learning rate according to a user-defined function.

**Important arguments:**

 - `optimizer` → optimizer whose learning rate will be changed.
 - `lr_lambda` → function that determines the learning-rate multiplier for each epoch/step.
 - `last_epoch` → index of the last completed epoch; -1 means start from the initial learning rate.

**Example:**
```python
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: 0.95 ** epoch
)

scheduler.step()
```
**Remember:**
The scheduler changes the learning rate over time according to lr_lambda.

**For example:**
```python
lr_lambda=lambda epoch: 0.95 ** epoch
```
means the learning rate is multiplied by 0.95 each epoch.

---

## `torch.randn()`

**What it does:**
Creates a tensor filled with random numbers sampled from a standard normal distribution (mean `0`, standard deviation `1`).

**Important arguments:**
- `*size` → dimensions/shape of the tensor.
- `device` → device where the tensor is created (`cpu`, `cuda`, etc.).
- `dtype` → data type of the tensor.

**Example:**
```python
x = torch.randn(3, 4)
```
**Remember:**
Useful for creating random input data or initializing values.

---

## `torch.no_grad()`

**What it does:**
Disables gradient tracking inside a block of code, typically during validation or inference.

**Example:**
```python
with torch.no_grad():
    output = model(x)
```
**Remember:**
Saves memory and computation when gradients are not needed. Tells PyTorch "No need to save calculations for backpropagation."

---

## `torch.optim.AdamW()`

**What it does:**
Creates an AdamW optimizer, which updates model parameters using adaptive learning rates and decoupled weight decay.

**Important arguments:**

- `params` → model parameters to optimize.
- `lr` → learning rate controlling the size of parameter updates.
- `weight_decay` → strength of weight decay regularization.
- `betas` → coefficients controlling the moving averages of gradients and squared gradients.
- `eps` → small value used for numerical stability.

**Example:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```
**Remember:**
lr controls how much weights change; weight_decay helps regularize the model.

---

## `F.cross_entropy()`

**What it does:**
Computes cross-entropy loss for classification by comparing predicted logits with the target class labels.

**Important arguments:**

- `input` → model's raw output/logits, usually shape [batch_size, num_classes].
- `target` → correct class indices, usually shape [batch_size].
- `weight` → optional weights for different classes.
- `reduction` → how individual losses are combined: 'mean', 'sum', or 'none'.

**Example:**
```python
loss = F.cross_entropy(output, target)
```
**Remember:**
The input should normally be raw logits, not probabilities; cross_entropy() applies the required normalization internally.

---

## `Tensor.to()`

**What it does:**
Moves a tensor to a different device and/or converts it to a different data type.

**Important arguments:**

- `device` → target device, e.g. cpu or cuda.
- `dtype` → target data type, e.g. torch.float32.
- `non_blocking` → allows asynchronous transfer when supported.

**Example:**
```python
x = x.to(device)
x = x.to(torch.float32)
```
**Remember:**
Very common for making sure model inputs and the model are on the same device.

---

## `optimizer.zero_grad()`

**What it does:**
Resets the gradients of the model parameters to zero before calculating gradients for the next training step.

**Important arguments:**
- `set_to_none` → if `True`, sets gradients to `None` instead of zero; can improve performance and reduce memory usage.

**Example:**
```python
optimizer.zero_grad()
loss.backward()
```
**Remember:**
PyTorch accumulates gradients by default, so they normally need to be cleared before each update.

---

## `loss.backward()`

**What it does:**
Computes the gradients of the loss with respect to the model's parameters using backpropagation.

**Important arguments:**

- `retain_graph` → keeps the computation graph after the backward pass if another backward pass is needed.
- `create_graph` → builds a graph of the gradient computation, useful for higher-order derivatives.

**Example:**
```python
loss.backward()
```
**Remember:**
backward() calculates the gradients; it does not update the weights.

## `optimizer.step()`

**What it does:**
Updates the model parameters using the gradients calculated by loss.backward().

**Important arguments:**

- `closure` → optional function that reevaluates the model and loss; mainly relevant to optimizers such as LBFGS.

**Example:**
```python
loss.backward()
optimizer.step()
```
**Remember:**
Typical training sequence:

zero_grad() → backward() → step()

---

## `mlp.load_state_dict(best_model_state)`

**What it does:**
Loads a saved set of model parameters and buffers into the mlp model.

**Important arguments:**

- `state_dict` → dictionary containing the saved model parameters and buffers.
- `strict` → if True, requires the keys in the saved state to exactly match the model.

**Example:**
```python
mlp.load_state_dict(best_model_state)
```
**Remember:**
Commonly used to restore the best model checkpoint, for example the model with the best validation performance.

## `mlp.eval()`

**What it does:**
Switches the model to evaluation mode.

**Example:**
```python
mlp.eval()

with torch.no_grad():
    output = mlp(x)
```
**Remember:**
Changes the behaviour of layers such as Dropout and BatchNorm for validation/testing. It does not disable gradients by itself.

---

## `import torch_geometric.nn as pyg_nn`

**What it does:**
Imports PyTorch Geometric's neural-network module, which provides GNN layers and graph-specific operations.

**Important:**
- `pyg_nn` → alias used to access the module.

**Example:**
```python
pyg_nn.GCNConv(...)
```

**Remember:**
Contains neural-network layers designed specifically for graph data.

---

## `GCNConv`

**What it does:**
Graph Convolutional Network layer that updates node representations using information from neighbouring nodes.

**Important arguments:**
- `in_channels` → number of input features per node.
- `out_channels` → number of output features per node.
- `improved` → uses an alternative normalization scheme when `True`.
- `add_self_loops` → whether to add self-connections.
- `bias` → whether to learn a bias parameter.

**Example:**
```python
conv = GCNConv(32, 64)
```

**Remember:**
`32 → 64` means each node's 32 features are transformed into 64 features using its neighbourhood.

---

## `GINConv`

**What it does:**
Graph Isomorphism Network convolution that aggregates neighbouring node features and applies a neural network to the result.

**Important arguments:**
- `nn` → neural network used to transform the aggregated features.
- `eps` → controls the weighting of the central node's features.
- `train_eps` → whether `eps` is learnable.

**Example:**
```python
conv = GINConv(nn.Sequential(nn.Linear(32, 64), nn.ReLU()))
```

**Remember:**
GIN uses an MLP after aggregating information from a node and its neighbours.

---

## `global_mean_pool`

**What it does:**
Aggregates node features into one graph-level representation by calculating the mean of node features.

**Important arguments:**
- `x` → node feature matrix.
- `batch` → tensor indicating which graph each node belongs to.

**Example:**
```python
graph_embedding = global_mean_pool(x, batch)
```

**Remember:**
Many nodes → **one vector per graph** using the mean.

---

## `global_add_pool`

**What it does:**
Aggregates node features into one graph-level representation by summing the features of all nodes in each graph.

**Important arguments:**
- `x` → node feature matrix.
- `batch` → tensor assigning each node to a graph.

**Example:**
```python
graph_embedding = global_add_pool(x, batch)
```

**Remember:**
Many nodes → **one vector per graph** using the sum.

---

## `global_max_pool`

**What it does:**
Aggregates node features into one graph-level representation by taking the maximum value for each feature across the nodes.

**Important arguments:**
- `x` → node feature matrix.
- `batch` → tensor assigning each node to a graph.

**Example:**
```python
graph_embedding = global_max_pool(x, batch)
```

**Remember:**
Many nodes → **one vector per graph** using the maximum value of each feature.

---

## `spmm`

**What it does:**
Performs sparse matrix multiplication, allowing multiplication involving sparse graph representations.

**Important arguments:**
- `src` → sparse matrix.
- `other` → matrix/tensor to multiply with the sparse matrix.

**Example:**
```python
result = spmm(adj, x)
```

**Remember:**
Useful for efficient operations involving large sparse graph adjacency matrices.

---

## `nn.init.normal_()`

**What it does:**
Initializes tensor values by sampling weights from a **Gaussian (normal) distribution**.

**Important arguments:**
- `tensor` → tensor to initialize.
- `mean` → mean of the distribution.
- `std` → standard deviation of the distribution.

**Example:**
```python
nn.init.normal_(t, mean=0.0, std=0.02)
```

**Remember:**
**Normal = simple random initialization** using a Gaussian distribution.

---

## `nn.init.uniform_()`

**What it does:**
Initializes tensor values by sampling weights from a **uniform distribution** within a specified range.

**Important arguments:**
- `tensor` → tensor to initialize.
- `a` → lower bound.
- `b` → upper bound.

**Example:**
```python
nn.init.uniform_(t, a=-0.1, b=0.1)
```

**Remember:**
**Uniform = simple random initialization with a bounded range.**

---

## `nn.init.xavier_normal_()`

**What it does:**
Initializes weights using a **normal distribution whose variance is determined by the input and output dimensions** of the layer.

**Important arguments:**
- `tensor` → tensor to initialize.
- `gain` → scaling factor applied to the initialization.

**Example:**
```python
nn.init.xavier_normal_(layer.weight)
```

**Remember:**
**Xavier Normal = normal distribution + layer dimensions → helps keep activation/gradient scales stable.**

---

## `nn.init.xavier_uniform_()`

**What it does:**
Initializes weights using the **Xavier method with a uniform distribution**, with the range determined by the layer's input/output dimensions.

**Important arguments:**
- `tensor` → tensor to initialize.
- `gain` → scaling factor applied to the initialization.

**Example:**
```python
nn.init.xavier_uniform_(layer.weight)
```

**Remember:**
**Xavier Uniform = same goal as Xavier Normal, but uses a uniform distribution instead.**

---

## `nn.init.orthogonal_()`

**What it does:**
Initializes weights using an **orthogonal matrix**, where the weight vectors are mutually orthogonal.

**Important arguments:**
- `tensor` → tensor to initialize.
- `gain` → scaling factor applied to the initialization.

**Example:**
```python
nn.init.orthogonal_(layer.weight)
```

**Remember:**
**Orthogonal = aims to preserve vector norms through transformations**, helping information remain stable as it passes through layers.

---

## `nn.Sequential()`

**What it does:**
Creates a container that applies a sequence of neural-network modules in order.

**Important arguments:**
- `*args` → layers/modules to execute sequentially.

**Example:**
```python
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
```

**Remember:**
Input → Layer 1 → Layer 2 → ... → Output.

---

## `nn.Parameter()`

**What it does:**
Creates a tensor that is automatically treated as a learnable model parameter.

**Important arguments:**
- `data` → initial tensor values.
- `requires_grad` → whether gradients should be calculated for the parameter.

**Example:**
```python
self.weight = nn.Parameter(torch.randn(10, 10))
```

**Remember:**
Parameters assigned to an `nn.Module` are automatically included in `model.parameters()`.

---

## `get_init_fn(config)`

**What it does:**
Likely a custom function from the repository that returns an initialization function based on the provided configuration.

**Important argument:**
- `config` → configuration containing the initialization settings to use.

**Example:**
```python
init_fn = get_init_fn(config)
```

**Remember:**
Unlike the functions above, `get_init_fn()` is **project-specific**, so its exact behaviour depends on its implementation in the repository.

---

## `torch.empty()`

**What it does:**
Creates a tensor with a specified shape **without initializing its values**.

**Important arguments:**
- `size` → shape/dimensions of the tensor.
- `dtype` → data type of the tensor.
- `device` → device where the tensor is created (`cpu`, `cuda`, etc.).

**Example:**
```python
x = torch.empty(3, 4)
```

**Remember:**
The values are **uninitialized** and may contain arbitrary memory values. Use it when you plan to fill the tensor afterwards.

---

## `torch.einsum()`

**What it does:**
Performs tensor operations such as multiplication, dot products, and summation using **Einstein summation notation**.

**Important arguments:**
- `equation` → string describing how input dimensions are combined.
- `operands` → tensors involved in the operation.

**Example:**
```python
result = torch.einsum("ij,jk->ik", A, B)
```

This performs matrix multiplication: `A @ B`.

**Remember:**
The letters represent tensor dimensions:
- Same letter → dimensions are multiplied/summed according to the equation.
- `->` → specifies the dimensions of the output.
- `...` → represents any number of batch dimensions.

---

## `field(default_factory=...)`

**What it does:**
Defines a default value for a field in a Python `dataclass`, with `default_factory` creating a **new value for each instance**.

**Important arguments:**
- `default_factory` → function called to create the default value.
- `init` → whether the field is included in the generated `__init__()` method.
- `repr` → whether the field appears in the object's representation.

**Example:**
```python
field(default_factory=lambda: [[64], [64, 64]])`
```
Here:
- `lambda: [[64], [64, 64]]` → function that creates the default list.
- Each dataclass instance gets its **own separate list**.

**Remember:**
`default_factory` is especially useful for **mutable defaults** such as lists and dictionaries, avoiding the same object being shared between instances.

---