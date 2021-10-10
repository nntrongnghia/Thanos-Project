import torch
from torchmetrics.classification import precision_recall
import wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from thanos.model import GestureTransformer
from thanos.trainers.config import BaseTrainConfig
from thanos.trainers.criterion import Criterion
from sklearn.metrics import confusion_matrix
from torchmetrics import Accuracy, Recall, Precision, ConfusionMatrix

class LitGestureTransformer(pl.LightningModule):
    def __init__(self, config:BaseTrainConfig,**kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.model = GestureTransformer(**config.model_config())
        self.criterion = Criterion(**config.criterion_config())
        self.val_acc = Accuracy()
        self.val_precision_per_class = Precision(
            num_classes=self.model.num_classes,
            average=None)
        self.val_recall_per_class = Recall(
            num_classes=self.model.num_classes,
            average=None)

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
        self.val_precision_per_class.update(preds, labels)
        self.val_recall_per_class.update(preds, labels)
        return total_loss

    def validation_epoch_end(self, outputs):
        # Log confusion matrix
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
        # Log other metrics
        self.log("val/accuracy", self.val_acc.compute())
        precision_per_class = self.val_precision_per_class.compute()
        recall_per_class = self.val_recall_per_class.compute()
        for i in range(self.model.num_classes):
            self.log("val/precision_" + str(i), precision_per_class[i])
            self.log("val/recall_" + str(i), recall_per_class[i])



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

    def log_data(self, data, prefix=""):
        for key, val in data.items():
            self.log(f"{prefix}/{key}", val)