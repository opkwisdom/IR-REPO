import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import (
    set_seed
)
from datamodule import ???
from model import ???

logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="llm2vec_train")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Save training configuration
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    config_output_path = f"{cfg.ckpt_dir}/settings.yaml"
    OmegaConf.save(config=cfg, f=config_output_path)

    # Prepare dataloader

    # Initialize model

    # Setup callbacks (step-based)
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        monitor='val_mrr_10',
        filename='dpr-{step:06d}-{val_mrr_10:.4f}',
        save_top_k=3,
        mode='max',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_cb, lr_monitor]

    # Setup logger
    wandb_logger = WandbLogger(
        project="Dense_Retrieval_Reproduce",
        group=cfg.exp_model,
        name=cfg.exp_name,
        tags=[cfg.exp_model, cfg.dataset.name, cfg.model.model_type]
    )

    # Setup Trainer

if __name__ == "__main__":
    main()