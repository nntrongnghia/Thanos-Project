import torch
import pytorch_lightning as pl
from thanos.model import build_detector

class LitDetector(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = build_detector()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)