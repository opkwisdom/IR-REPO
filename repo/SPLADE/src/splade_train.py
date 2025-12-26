from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import torch
from transformers import AutoTokenizer

import pytorch_lightning as pl
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
        filename='dpr-{epoch:02d}-{val_mrr_10:.4f}',
        save_top_k=3,
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_cb, lr_monitor]

    # Setup logger
    wandb_logger = WandbLogger(
        project="Dense_Retrieval_Reproduce",
        group=cfg.exp_model,
        name=cfg.exp_name,
        tags=[cfg.exp_model, cfg.dataset.name]
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        # strategy="ddp_find_unused_parameters_true"
        strategy="auto"
    )
    # trainer.fit(splade_module, splade_datamodule)
    trainer.validate(splade_module, splade_datamodule)

if __name__ == "__main__":
    main()