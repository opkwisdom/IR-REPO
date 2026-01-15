import hydra
import logging
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import (
    set_seed
)
from datamodule import LLM2VECDataModule
from model import LLM2VECEncoder, LLM2VECLightningModule

logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="llm2vec_train")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Load model & tokenizer