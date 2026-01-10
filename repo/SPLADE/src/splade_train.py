from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import torch
import os
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import (
    set_seed
)
from datamodule import SpladeDataModule
from model import SpladeEncoder, SpladeLightningModule

logger = logging.getLogger(__file__)

@hydra.main(version_base=None, config_path="../conf", config_name="splade_train")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Save training configuration
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    config_output_path = f"{cfg.ckpt_dir}/settings.yaml"
    OmegaConf.save(config=cfg, f=config_output_path)

    # Prepare dataloader
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    splade_datamodule = SpladeDataModule(cfg, tokenizer)

    # Initialize model
    model = SpladeEncoder(cfg.model)
    splade_module = SpladeLightningModule(cfg, model)

    # Setup callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        monitor='val_mrr_10',
        filename='splade-{step:06d}-{val_mrr_10:.4f}',
        save_top_k=3,
        mode='max',
        save_last=True,
        every_n_train_steps=cfg.train.val_check_interval
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
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
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=cfg.train.val_check_interval,
        check_val_every_n_epoch=None,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
        # strategy="auto"
    )
    
    # Resume from checkpoint if specified
    last_ckpt_path = os.path.join(cfg.ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        logger.info(f"Resuming training from checkpoint: {last_ckpt_path}")
        trainer.fit(splade_module, splade_datamodule, ckpt_path=last_ckpt_path)
    else:
        logger.info("Starting training from scratch.")
        trainer.fit(splade_module, splade_datamodule)
    # trainer.validate(splade_module, splade_datamodule)

if __name__ == "__main__":
    main()