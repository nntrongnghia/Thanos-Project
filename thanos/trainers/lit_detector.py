import torch
import wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from thanos.model import GestureTransformer
from thanos.trainers.config import BaseTrainConfig
from thanos.trainers.criterion import Criterion
from sklearn.metrics import confusion_matrix

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
        total_loss, data = self.criterion(m_outputs, labels, validation=True)
        self.log_data(total_loss, data, prefix="val")
        return total_loss

    def validation_epoch_end(self, outputs):
        val_preds = torch.cat(self.criterion.val_preds).detach().cpu().numpy()
        val_labels = torch.cat(self.criterion.val_labels).detach().cpu().numpy()
        self.logger.experiment.log(
            {
                "val/confusion_matrix": 
                    wandb.plot.confusion_matrix(
                        y_true=val_labels,
                        preds=val_preds,
                        class_names=self.config.class_names),
                "epoch": self.current_epoch
            })
        print(confusion_matrix(val_labels, val_preds))
        self.criterion.clear_validation_buffers()


    def configure_optimizers(self):
        lr = self.config.lr
        if isinstance(lr, float):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            # lr_scheduler = {
            #     'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,], gamma=0.1),
            #     'name': 'lr step scheduler'
            # }
            # return [optimizer], [lr_scheduler]
            return optimizer
        else:
            raise NotImplementedError()

    def log_data(self, total_loss, data, prefix=""):
        self.log(f"{prefix}/total_loss", total_loss)
        for key, val in data.items():
            self.log(f"{prefix}/{key}", val)