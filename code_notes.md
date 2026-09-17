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