import torch
import torch.nn.functional as F

def mse_loss_onehot(input: torch.Tensor, target_index: torch.Tensor, reduction: str = "none", num_classes: int = 10):
    target_index = target_index.long()
    target_onehot = F.one_hot(target_index, num_classes=num_classes).float()
    return F.mse_loss(input, target_onehot, reduction=reduction)