import argparse
import os

import pytorch_lightning as pl
import wandb

from thanos.trainers import load_config
from thanos.trainers.lit_detector import LitGestureTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config py file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    args = parser.parse_args()
    config = load_config(args.config)
    lit_model = LitGestureTransformer.load_from_checkpoint(args.checkpoint, config=config)
    trainer = pl.Trainer(**config.trainer_config())
    trainer.validate(lit_model, config.val_dataloader())
