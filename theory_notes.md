# GNNs Don't Need Backprop — Theory Notes

## 1. Graph Isomorphism Network (GIN)

**Definition:**  
A GNN architecture designed to distinguish graph structures through message passing.

**Key idea:**  
Node representations are updated using information from neighbouring nodes.

**In the paper:**  
One of the two GNN encoders tested (alongside GCN).

---

## 2. Graph Convolutional Network (GCN)

**Definition:**  
A GNN architecture that aggregates information from neighbouring nodes to create node embeddings.

**Key idea:**  
Repeated message passing captures information about the graph's structure.

**In the paper:**  
One of the two encoder architectures tested (alongside GIN).

---

## 3. Triangle Inequality

**Definition:**  
A distance function must satisfy:

$$
d(A,C) \leq d(A,B) + d(B,C)
$$

**Key idea:**  
The direct distance cannot be greater than going through an intermediate point.

**In the paper:**  
One of the properties required by the pseudometrics.

---

## 4. Pseudometric

**Definition:**  
A function measuring distance between objects that satisfies symmetry, non-negativity, the triangle inequality, and $d(x,x)=0$.

**Key idea:**  
Different objects can have distance $0$.

**In the paper:**  
$d_\phi$ measures embedding distance; $d_y$ measures label difference.

---

## 5. Weisfeiler–Leman (WL) Graph Isomorphism Test

**Definition:**  
An iterative procedure that tests whether graphs can be distinguished based on their structural patterns.

**Key idea:**  
Graphs indistinguishable by WL may also receive the same GNN embedding.

**In the paper:**  
Explains why $d_\phi(G,H)=0$ does not necessarily mean that $G$ and $H$ are isomorphic.

---

## 6. Alignment Score ($ALI_k$)

**Definition:**  
Measures how well the GNN embedding space reflects the graph labels.

**Key idea:**  
If nearest neighbours in embedding space usually have the same label → high ALI.

**In the paper:**  
Used to select the best randomly initialized GNN encoder.

---

## 7. AdamW Optimizer

**Definition:**  
An optimizer used to update neural-network weights during training.

**Key idea:**  
AdamW is Adam with decoupled weight decay.

**In the paper:**  
Used to train the GNN baseline and the MLP decoder.

---

## 8. Macro-averaged Vargha–Delaney $A_{12}$

**Definition:**  
A non-parametric effect-size measure comparing the performance of two methods.

**Interpretation:**
- $A_{12}=0.5$ → no difference
- $A_{12}>0.5$ → first method tends to perform better
- $A_{12}<0.5$ → second method tends to perform better

**Macro-averaged:**  
$A_{12}$ is calculated for each dataset–architecture combination, then all combinations receive equal weight.

**In the paper:**  
$A_{12}=0.48$, interpreted as no meaningful difference between the two training methods.

---

## 9. 5-Fold Stratified Cross-Validation (80/10/10)

**Definition:**  
The dataset is divided into 5 folds while preserving class proportions.

**Process:**  
Each fold is used once as the test set; the remaining data provides training and validation data.

**Why 3 sets?**
- **Training:** learn model parameters
- **Validation:** choose hyperparameters
- **Testing:** evaluate the final model on unseen data

**In the paper:**  
Hyperparameters are selected using validation accuracy, then final performance is measured on the test set.

---

## 10. ReLU Activation Function

**Definition:**  
ReLU (Rectified Linear Unit) is an activation function defined as:

$$
\mathrm{ReLU}(x) = \max(0,x)
$$

**Key idea:**  
Negative values become $0$, while positive values remain unchanged.

**Purpose:**  
Introduces **non-linearity**, allowing neural networks to learn complex patterns.

---

## 11. Dropout Rate

**Definition:**  
The dropout rate is the **probability of randomly setting a neuron’s output to zero during training**.

**Key idea:**  
For example, a dropout rate of $0.2$ means that approximately 20% of activations are randomly dropped during each training step.

**Purpose:**  
Acts as **regularization**, reducing overfitting by preventing the network from relying too heavily on particular neurons.

**Note:**  
Dropout is normally disabled during evaluation/testing.

---

## 12. Batch Normalization (Batch Norm)

**Definition:**  
A technique that normalizes layer activations within each mini-batch to stabilize and accelerate neural network training. It also uses learnable parameters to scale and shift the normalized activations.

**Key idea:**  
It helps keep the values passed between layers at a more stable scale during training.

**Purpose:**  
Can make training **more stable and faster** and can sometimes improve generalization.

**Typical position:**

$$
\text{Layer} \rightarrow \text{Batch Norm} \rightarrow \text{ReLU}
$$

---

## 13. Cross-Entropy Loss

**Definition:**
Measures the difference between the predicted and true class distributions:
$$
L = -\log(p_y)
$$
where $p_y$ is the predicted probability of the true class. For a batch:
$$
L = -\frac{1}{N}\sum_{n=1}^{N}\log(p_{n,y_n})
$$

**Key idea:**
Higher probability assigned to the correct class → lower loss; lower probability → higher loss.

**Purpose:**
Guides the model to assign higher probabilities to the correct classes during training.

**Note:**
In PyTorch, `F.cross_entropy()` takes raw logits and internally applies the required softmax/log operations.