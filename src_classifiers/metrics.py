import torch
import torch.nn.functional as F

def compute_ali(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5, block_size: int = 1000, mode: str = 'classification', metric: str = 'l2') -> torch.Tensor:
    """Compute ALI scores for a batch of embeddings."""
    
    if embeddings.shape[0] == labels.shape[0]: 
        embeddings = embeddings.permute(1, 0, 2)
        
    M, N, D = embeddings.shape
    device = embeddings.device
    
    if metric == 'cosine':
        embeddings_proc = F.normalize(embeddings, p=2, dim=2, eps=1e-8)
    elif metric in ['l2', 'l1']:
        embeddings_proc = embeddings
    else:
        raise ValueError(f"Unknown metric '{metric}'. Supported: 'cosine', 'l2', 'l1'")
    
    if mode == 'classification':
        classes, counts = torch.unique(labels, return_counts=True)
        probs = counts.float() / N
        B_k = (probs * (1 - probs)).sum()
    else:
        labels_float = labels.float().view(-1, 1)
        B_k = torch.cdist(labels_float, labels_float, p=1).mean()

    total_distance_sum = torch.zeros(M, device=device, dtype=torch.float32)
    
    for start_idx in range(0, N, block_size):
        end_idx = min(start_idx + block_size, N)
        current_block_len = end_idx - start_idx
        
        query_emb = embeddings_proc[:, start_idx:end_idx, :]
        
        if metric == 'cosine':
            sim_block = torch.bmm(query_emb, embeddings_proc.transpose(1, 2))
            largest = True
            fill_val = -float('inf')
        elif metric == 'l2':
            sim_block = torch.cdist(query_emb, embeddings_proc, p=2)
            largest = False
            fill_val = float('inf')
        elif metric == 'l1':
            sim_block = torch.cdist(query_emb, embeddings_proc, p=1)
            largest = False
            fill_val = float('inf')
            
        block_rows = torch.arange(current_block_len, device=device)
        global_cols = torch.arange(start_idx, end_idx, device=device)
        sim_block[:, block_rows, global_cols] = fill_val
        
        _, neighbor_indices = sim_block.topk(k, dim=2, largest=largest)
        
        neighbor_labels = labels[neighbor_indices] 
        
        center_labels_block = labels[start_idx:end_idx]
        center_labels = center_labels_block.view(1, -1, 1).expand(M, current_block_len, k)
        
        if mode == 'classification':
            dists = (neighbor_labels != center_labels).float()
        else:
            dists = (neighbor_labels - center_labels).abs()
        
        total_distance_sum += dists.sum(dim=(1, 2))

    A_k = total_distance_sum / (N * k)
    return B_k - A_k
