import os
import datetime
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from thanos.trainers import BaseTrainConfig
from thanos.dataset import IPN, IPN_HAND_ROOT, one_hot_label_transform
from thanos.trainers.data_augmentation import (
    get_temporal_transform_fn, 
    get_train_spatial_transform_fn, 
    get_val_spatial_transform_fn)

class DefaultConfig(BaseTrainConfig):
    PROJECT_NAME = "GestureTransformer"
    EXPE_NAME = "default_config"

    def __init__(self):
        self.config_path = __file__
        self.logger = WandbLogger(
            name="{}_{:%B-%d-%Y-%Hh-%M}".format(self.EXPE_NAME, datetime.datetime.now()),
            project=self.PROJECT_NAME,
            offline=True)

        # === train config ===
        self.accumulate_grad_batches = 4
        self.batch_size = 4
        self.lr = 1e-4
        self.class_weights = torch.tensor(
            [0.1429, 0.2000, 0.2000, 1.0000, 
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            device=torch.device("cuda")
        )

        # === Dataset ===
        self.num_classes = 14
        self.input_duration = 22 # frames
        self.temporal_stride = 2
        self.ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
        self.train_ipn = IPN(IPN_HAND_ROOT, self.ann_path, "training",
            temporal_stride=self.temporal_stride,
            spatial_transform=get_train_spatial_transform_fn(), 
            temporal_transform=get_temporal_transform_fn(self.input_duration),
            target_transform=one_hot_label_transform,
            sample_duration=self.temporal_stride*self.input_duration,
            n_samples_for_each_video=2
            )
        self.val_ipn = IPN(IPN_HAND_ROOT, self.ann_path, "validation",
            temporal_stride=self.temporal_stride,
            spatial_transform=get_val_spatial_transform_fn(), 
            temporal_transform=get_temporal_transform_fn(self.input_duration),
            target_transform=one_hot_label_transform)

        # save this config file to wandb cloud

    def model_config(self):
        return {
            "backbone": "resnet18",
            "num_classes": self.num_classes,
            "encoder_dim": 256,
            "vqk_dim": 128,
            "encoder_fc_dim": 256,
            "n_encoder_heads": 6,
            "n_encoders": 6
        }
    

    def trainer_config(self):
        return {
            "accumulate_grad_batches": self.accumulate_grad_batches, 
            "logger": self.logger,
            "gpus": 1,
            "log_every_n_steps": 20
        }

    def criterion_config(self):
        return {
            "num_classes": self.num_classes,
            "class_weights": self.class_weights
        }

    def train_dataloader(self):
        return DataLoader(self.train_ipn, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ipn, batch_size=self.batch_size, shuffle=False, num_workers=2)

config = DefaultConfig()