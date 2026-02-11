import torch
import torch.nn.functional as F

def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Compute the supervised contrastive loss.
    
    Args:
        features (Tensor): Embeddings of shape (batch_size, feature_dim). Should be normalized.
        labels (Tensor): Integer labels of shape (batch_size,).
        temperature (float): Scaling factor.
        
    Returns:
        loss (Tensor): Scalar loss.
    """
    device = features.device
    batch_size = features.size(0)
    
    # Expand labels to compare each pair
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # mask[i, j] = 1 if labels[i]==labels[j]
    
    # Compute cosine similarity scaled by temperature
    logits = torch.matmul(features, features.T) / temperature

    # For numerical stability, subtract the max logit per row
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    
    # Remove self-contrast (i.e., i==j)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
    mask = mask * logits_mask

    # Compute log probabilities
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    
    # For each anchor, average log-probabilities over the positive samples
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    
    # Loss: Negative of the average of the above values
    loss = -mean_log_prob_pos.mean()
    return loss

# Example usage:
# Suppose features is a normalized tensor of shape (batch_size, feature_dim)
# and labels is a tensor of shape (batch_size,) indicating the class of each sample.
features = torch.randn(8, 128)  # example embeddings; in practice, use your model's output and normalize them
features = F.normalize(features, dim=1)
labels = torch.tensor([0, 1, 0, 1, 0, 2, 2, 0])  # e.g., image 0,2,4,7 share the same label

loss_value = supervised_contrastive_loss(features, labels)
print("Supervised Contrastive Loss:", loss_value.item())
