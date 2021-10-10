import os
import datetime
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from thanos.trainers import BaseTrainConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from thanos.dataset import IPN, IPN_HAND_ROOT, one_hot_label_transform
from thanos.trainers.data_augmentation import (
    get_temporal_transform_fn, 
    get_train_spatial_transform_fn, 
    get_val_spatial_transform_fn)
from pytorch_lightning.callbacks import LearningRateMonitor

class DefaultConfig(BaseTrainConfig):
    PROJECT_NAME = "GestureTransformer"
    EXPE_NAME = "default_config"

    def __init__(self):
        self.config_path = __file__
        self.expe_name = "{}_{:%B-%d-%Y-%Hh-%M}".format(self.EXPE_NAME, datetime.datetime.now())
        self.logger = WandbLogger(
            name=self.expe_name,
            project=self.PROJECT_NAME,
            # offline=True # for debug
            )

        # === train config ===
        self.lr_monitor = LearningRateMonitor(logging_interval='step')
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val/total_loss",
            save_top_k=3)
        self.callbacks = [self.lr_monitor, self.checkpoint_callback]
        self.default_root_dir = os.path.join(os.getcwd(), self.PROJECT_NAME, self.expe_name)
        self.accumulate_grad_batches = 4
        self.batch_size = 4
        self.class_weights = torch.tensor(
            # [7.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.5, 1.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            device=torch.device("cuda"))
        self.lr = 1e-4
        self.lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)
        # === Dataset ===
        self.num_classes = 14
        self.class_names = IPN.CLASSES
        self.input_duration = 22 # frames
        self.temporal_stride = 2
        self.ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
        self.train_ipn = IPN(IPN_HAND_ROOT, self.ann_path, "training",
            temporal_stride=self.temporal_stride,
            spatial_transform=get_train_spatial_transform_fn(), 
            temporal_transform=get_temporal_transform_fn(self.input_duration),
            target_transform=one_hot_label_transform,
            sample_duration=self.temporal_stride*self.input_duration,
            n_samples_for_each_video=1
            )
        self.val_ipn = IPN(IPN_HAND_ROOT, self.ann_path, "validation",
            temporal_stride=self.temporal_stride,
            spatial_transform=get_val_spatial_transform_fn(), 
            temporal_transform=get_temporal_transform_fn(self.input_duration, training=False),
            target_transform=one_hot_label_transform)

        # save this config file to wandb cloud

    def model_config(self):
        return {
            "backbone": "resnet18",
            "num_classes": self.num_classes,
            "encoder_dim": 512,
            "vqk_dim": 64,
            "encoder_fc_dim": 512,
            "n_encoder_heads": 8,
            "n_encoders": 6,
            "return_aux": True
        }
    

    def trainer_config(self):
        return {
            "accumulate_grad_batches": self.accumulate_grad_batches, 
            "logger": self.logger,
            "gpus": 1,
            "log_every_n_steps": 20,
            "gradient_clip_val": 1.0,
            "callbacks": self.callbacks,
            "default_root_dir": self.default_root_dir,
            # "track_grad_norm": 2, # for debug
            # "limit_train_batches": 0.05 # for debug
        }

    def criterion_config(self):
        return {
            "num_classes": self.num_classes,
            "class_weights": self.class_weights
        }

    def train_dataloader(self):
        return DataLoader(self.train_ipn, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ipn, batch_size=self.batch_size, shuffle=False, num_workers=1)

config = DefaultConfig()