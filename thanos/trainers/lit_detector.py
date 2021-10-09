import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from thanos.model import GestureTransformer
from thanos.trainers.config import BaseTrainConfig
from thanos.trainers.criterion import Criterion

class LitGestureTransformer(pl.LightningModule):
    def __init__(self, config:BaseTrainConfig,**kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.model = GestureTransformer(**config.model_config())
        self.criterion = Criterion(**config.criterion_config())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        m_outputs = self.model(sequences)
        loss = self.criterion(m_outputs, labels)
        self.log("train/sigmoid_focal_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        m_outputs = self.model(sequences)
        loss = self.criterion(m_outputs, labels)
        self.log("val/sigmoid_focal_loss", loss)
        return loss

    def configure_optimizers(self):
        lr = self.config.lr
        if isinstance(lr, float):
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError()