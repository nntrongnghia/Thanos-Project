import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from thanos.model import GestureTransformer
from thanos.trainers.focal_loss import sigmoid_focal_loss
from thanos.trainers.config import BaseTrainConfig

def criterion(probs, labels, class_weights=None):
    labels = labels.to(torch.float)
    # loss = sigmoid_focal_loss(probs, labels)*probs.shape[-1]
    loss = F.binary_cross_entropy_with_logits(probs, labels, class_weights)
    return loss

class LitGestureTransformer(pl.LightningModule):
    def __init__(self, config:BaseTrainConfig,**kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.model = GestureTransformer(**config.model_config())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences)
        loss = criterion(logits, labels, self.config.class_weights)
        self.log("train/sigmoid_focal_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences)
        loss = criterion(logits, labels, self.config.class_weights)
        self.log("val/sigmoid_focal_loss", loss)
        return loss

    def configure_optimizers(self):
        lr = self.config.lr
        if isinstance(lr, float):
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError()