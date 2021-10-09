import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from thanos.model import GestureTransformer
from thanos.trainers.focal_loss import sigmoid_focal_loss
from thanos.trainers.config import BaseTrainConfig

def criterion(probs, labels):
    labels = labels.to(torch.float)
    loss = sigmoid_focal_loss(probs, labels)
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
        loss = criterion(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self.model(sequences)
        loss = criterion(logits, labels)
        return loss

    def configure_optimizers(self):
        train_config = self.config.train_config()
        lr = train_config["lr"]
        if isinstance(lr, float):
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError()