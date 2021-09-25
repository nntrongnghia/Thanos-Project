import torch
import torch.nn.functional as F

def sigmoid_focal_loss(inputs:torch.Tensor, targets:torch.Tensor, alpha: float = 0.25, gamma: float = 2)-> torch.Tensor:
    """Sigmoid focal loss for classification

    Parameters
    ----------
    inputs : torch.Tensor
        The predictions for each example.
    targets : torch.Tensor
        A float tensor with the same shape as inputs. Stores the binary
        classification label for each element in inputs
        (0 for the negative class and 1 for the positive class).
    alpha : float, optional
        (optional) Weighting factor in range (0,1) to balance
        positive vs negative examples. -1 for no weighting. By default 0.25.
    gamma : float, optional
        Exponent of the modulating factor (1 - p_t) to
        balance easy vs hard examples. By default 2
    Returns
    -------
    torch.Tensor
        Scalar
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()