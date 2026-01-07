from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import torch
import os
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import (
    set_seed,
)
from datamodule import ColBERTDataModule
from model import ColBERTEncoder, ColBERTLightningModule

logger = logging.getLogger(__file__)

@hydra.main(version_base=None, config_path="../conf", config_name="colbert_v1_train")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Save training configuration
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    config_output_path = f"{cfg.ckpt_dir}/settings.yaml"
    OmegaConf.save(config=cfg, f=config_output_path)

    # Prepare dataloader
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    # Add special tokens
    special_tokens = ["[Q]", "[D]"]
    tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    cfg.model.vocab_size = len(tokenizer)
    logger.info(f"Update vocab size to {cfg.model.vocab_size}")
    colbert_datamodule = ColBERTDataModule(cfg, tokenizer)


    # Initialize model
    model = ColBERTEncoder(cfg.model)
    model.resize_token_embeddings(len(tokenizer))
    colbert_module = ColBERTLightningModule(cfg, model, tokenizer)

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

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=-1,
        max_steps=cfg.train.max_steps,
        val_check_interval=cfg.train.val_check_interval,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        strategy="ddp_find_unused_parameters_true",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    trainer.fit(colbert_module, colbert_datamodule)

if __name__ == "__main__":
    main()