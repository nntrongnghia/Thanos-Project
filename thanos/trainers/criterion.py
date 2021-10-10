import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.val_labels = []
        self.val_preds = []

    def clear_validation_buffers(self):
        self.val_labels = []
        self.val_preds = []


    def forward(self, m_outputs, onehot_labels, validation=False):
        """
        Parameters
        ----------
        m_outputs: dict
            output dict of model
        labels: torch.Tensor
            one-hot labels, shape (B, nb_classes)
        validation: bool
            if True, confusion matrix will be accumulate 
            and some metrics will be added in `data`

        Returns
        -------
        total_loss: torch.Tensor
        data: dict
            useful data to log
        """
        logits = m_outputs["logits"] # (B, 14)
        onehot_labels = onehot_labels.to(torch.float)
        # === Losses ===
        total_loss = F.binary_cross_entropy_with_logits(logits, onehot_labels, self.class_weights)
        data = {"BCE_loss":total_loss.item()}
        if "aux" in m_outputs:
            for i, aux_logits in enumerate(m_outputs["aux"]):
                loss = F.binary_cross_entropy_with_logits(aux_logits, onehot_labels, self.class_weights)
                total_loss += loss
                data["BCE_loss_" + str(i)] = loss.item()
        # === Metrics ===
        if validation:
            preds = torch.argmax(logits, dim=-1)
            labels = torch.argmax(onehot_labels, dim=-1)
            self.val_preds.append(preds)
            self.val_labels.append(labels)
        return total_loss, data