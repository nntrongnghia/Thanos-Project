import os
import wandb
import pytorch_lightning as pl
import argparse

from thanos.trainers.lit_detector import LitGestureTransformer
from thanos.trainers import load_config



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config py file")
    args = parser.parse_args()
    config = load_config(args.config)
    lit_model = LitGestureTransformer(config)
    trainer = pl.Trainer(**config.trainer_config())
    trainer.fit(lit_model, config.train_dataloader(), config.val_dataloader())
    # save the config file to wandb cloud
    wandb.save(config.config_path)