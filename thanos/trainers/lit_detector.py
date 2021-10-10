import torch
import wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Recall, Precision, ConfusionMatrix

from thanos.model import GestureTransformer
from thanos.trainers.config import BaseTrainConfig
from thanos.trainers.criterion import Criterion
from thanos.trainers.log_utils import get_wandb_confusion_matrix_plot

class LitGestureTransformer(pl.LightningModule):
    def __init__(self, config:BaseTrainConfig,**kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.model = GestureTransformer(**config.model_config())
        self.criterion = Criterion(**config.criterion_config())
        self.val_acc = Accuracy()
        self.val_confusion_matrix = ConfusionMatrix(self.model.num_classes)

    def forward(self, x):
        return self.model(x)

    def inference(self, m_outputs):
        logits = m_outputs["logits"]
        return torch.argmax(logits, dim=-1)

    def training_step(self, batch, batch_idx):
        sequences, onehot_labels = batch
        m_outputs = self.model(sequences)
        total_loss, data = self.criterion(m_outputs, onehot_labels)
        self.log_data(data, prefix="train")
        return total_loss


    def validation_step(self, batch, batch_idx):
        sequences, onehot_labels = batch
        m_outputs = self.model(sequences)
        # === Losses
        total_loss, data = self.criterion(m_outputs, onehot_labels, validation=True)
        self.log_data(data, prefix="val")
        # === Metrics
        preds = self.inference(m_outputs)
        labels = torch.argmax(onehot_labels, dim=-1) 
        self.val_acc(preds, labels)
        self.val_confusion_matrix.update(preds, labels)
        return total_loss

    def validation_epoch_end(self, outputs):
        # Log confusion matrix
        confusion_matrix = self.val_confusion_matrix.compute().cpu().numpy()
        self.logger.experiment.log(
            {
                "val/confusion_matrix": 
                    get_wandb_confusion_matrix_plot(
                        confusion_matrix, 
                        self.config.class_names),
                "trainer/global_step": self.global_step
            })
        print(confusion_matrix)
        # Log accuracy
        self.log("val/accuracy", self.val_acc.compute())
        # Reset metrics
        self.val_confusion_matrix.reset()
        self.val_acc.reset()



    def configure_optimizers(self):
        lr = self.config.lr
        if isinstance(lr, float):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            lr_scheduler = {
                'scheduler': self.config.lr_scheduler_fn(optimizer),
                'name': 'lr_step_scheduler'
            }
            return [optimizer], [lr_scheduler]
            # return optimizer
        else:
            raise NotImplementedError()

    def log_data(self, data, prefix=""):
        for key, val in data.items():
            self.log(f"{prefix}/{key}", val)