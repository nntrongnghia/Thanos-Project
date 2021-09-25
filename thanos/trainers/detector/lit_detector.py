import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from thanos.model import GestureDetector
from thanos.trainers.focal_loss import sigmoid_focal_loss

def criterion(probs, labels):
    labels = labels[..., None].to(torch.float)
    loss = sigmoid_focal_loss(probs, labels)
    return loss

class LitDetector(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = GestureDetector()
        self.class_weight = [1, 3] # [no gesture, gesture]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences)
        loss = criterion(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences)
        loss = criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)