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