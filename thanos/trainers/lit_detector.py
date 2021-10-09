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
        total_loss, data = self.criterion(m_outputs, labels)
        self.log_data(total_loss, data, prefix="train")
        return total_loss


    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        m_outputs = self.model(sequences)
        total_loss, data = self.criterion(m_outputs, labels)
        self.log_data(total_loss, data, prefix="val")
        return total_loss

    def configure_optimizers(self):
        lr = self.config.lr
        if isinstance(lr, float):
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError()

    def log_data(self, total_loss, data, prefix=""):
        self.log(f"{prefix}/total_loss", total_loss)
        for key, val in data.items():
            self.log(f"{prefix}/{key}", val)