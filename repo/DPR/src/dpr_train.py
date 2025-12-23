from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import json
import os
import torch
import random
from dataclasses import asdict
from typing import List
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import (
    load_queries,
    load_qrels,
    set_seed,
    Query, Qrel, Triple
)
from datamodule import DPRDataModule
from model import DPREncoder, DPRLightningModule

logger = logging.getLogger(__file__)

@hydra.main(version_base=None, config_path="../conf", config_name="dpr_train")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Prepare dataloader
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    dpr_datamodule = DPRDataModule(cfg, tokenizer)

    # Initialize model
    model = DPREncoder(cfg.model)
    dpr_module = DPRLightningModule(cfg, model)

    # Setup callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        monitor='val_loss',
        filename='dpr-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_cb, lr_monitor]
    
    # Setup logger
    wandb_logger = WandbLogger(
        project="Dense_Retrieval_Reproduce",
        group=cfg.exp_model,
        name=cfg.exp_name,
        tags=[cfg.exp_model, cfg.data.name]
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.num_device,
        max_epochs=cfg.train.max_epochs,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        strategy="ddp_find_unused_parameters_true"
    )
    trainer.fit(dpr_module, dpr_datamodule)


if __name__ == "__main__":
    main()