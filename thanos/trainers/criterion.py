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
        loss = F.binary_cross_entropy_with_logits(logits, labels, self.class_weights)
        if "aux" in m_outputs:
            for logits in m_outputs["aux"]:
                loss += F.binary_cross_entropy_with_logits(logits, labels, self.class_weights)
        return loss