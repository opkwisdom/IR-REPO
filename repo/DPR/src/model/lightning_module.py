import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

from .models import DPREncoder


class DPRLightningModule(pl.LightningModule):
    def __init__(self, model: DPREncoder, learning_rate: float = 1e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)