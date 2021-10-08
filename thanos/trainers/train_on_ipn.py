import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from thanos.dataset import IPN, IPN_HAND_ROOT, one_hot_label_transform
from thanos.trainers.data_augmentation import (
    get_temporal_transform_fn, 
    get_train_spatial_transform_fn, 
    get_val_spatial_transform_fn)
from thanos.trainers.lit_detector import LitGestureTransformer


if __name__ == "__main__":

    ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    train_ipn = IPN(IPN_HAND_ROOT, ann_path, "training",
        temporal_stride=2,
        spatial_transform=get_train_spatial_transform_fn(), 
        temporal_transform=get_temporal_transform_fn(20),
        target_transform=one_hot_label_transform,
        n_samples_for_each_video=2)
    train_dataloader = DataLoader(train_ipn, batch_size=2, shuffle=True, num_workers=3)

    val_ipn = IPN(IPN_HAND_ROOT, ann_path, "validation",
        temporal_stride=2,
        spatial_transform=get_val_spatial_transform_fn(), 
        temporal_transform=get_temporal_transform_fn(20),
        target_transform=one_hot_label_transform)
    val_dataloader = DataLoader(val_ipn, batch_size=2, shuffle=False, num_workers=3)

    lit_model = LitGestureTransformer()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(lit_model, train_dataloader, val_dataloader)
