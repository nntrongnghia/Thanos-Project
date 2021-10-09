import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.confusion_matrix = torch.zeros((num_classes, num_classes))

    def forward(self, m_outputs, labels, validation=False):
        logits = m_outputs["logits"]
        labels = labels.to(torch.float)
        total_loss = F.binary_cross_entropy_with_logits(logits, labels, self.class_weights)
        data = {"BCE_loss":total_loss.item()}
        if "aux" in m_outputs:
            for i, logits in enumerate(m_outputs["aux"]):
                loss = F.binary_cross_entropy_with_logits(logits, labels, self.class_weights)
                total_loss += loss
                data["BCE_loss_" + str(i)] = loss.item()
        return total_loss, data