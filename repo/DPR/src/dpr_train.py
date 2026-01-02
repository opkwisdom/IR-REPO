from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import torch
from transformers import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import (
    set_seed,
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
    if getattr(cfg.train, 'max_epochs', None) is None:
        checkpoint_cb = ModelCheckpoint(
            dirpath=cfg.ckpt_dir,
            monitor='val_mrr_10',
            filename='dpr-{step:06d}-{val_mrr_10:.4f}',
            save_top_k=3,
            mode='max',
            save_last=True
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            dirpath=cfg.ckpt_dir,
            monitor='val_mrr_10',
            filename='dpr-{epoch:02d}-{val_mrr_10:.4f}',
            save_top_k=3,
            mode='max',
            save_last=True
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
    max_epochs = cfg.train.max_epochs if getattr(cfg.train, 'max_epochs', None) is not None else -1
    max_steps = cfg.train.max_steps if getattr(cfg.train, 'max_steps', None) is not None else -1
    val_check_interval = cfg.train.val_check_interval if getattr(cfg.train, 'val_check_interval', None) is not None else None
    assert not (max_epochs == -1 and max_steps == -1), "Either max_epochs or max_steps must be set."

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=max_epochs,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        strategy="ddp_find_unused_parameters_true",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    trainer.fit(dpr_module, dpr_datamodule)


if __name__ == "__main__":
    main()