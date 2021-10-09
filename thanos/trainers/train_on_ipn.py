import os
import torch
from torch.serialization import load
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse

from thanos.dataset import IPN, IPN_HAND_ROOT, one_hot_label_transform
from thanos.trainers.data_augmentation import (
    get_temporal_transform_fn, 
    get_train_spatial_transform_fn, 
    get_val_spatial_transform_fn)
from thanos.trainers.lit_detector import LitGestureTransformer
from thanos.trainers import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config py file")
    args = parser.parse_args()
    config = load_config(args.config)

    lit_model = LitGestureTransformer(config)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(lit_model, config.train_dataloader(), config.val_dataloader())
