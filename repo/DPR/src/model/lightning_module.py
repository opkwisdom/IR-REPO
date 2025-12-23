import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from omegaconf import DictConfig

from .models import DPREncoder


class DPRLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: DPREncoder):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = cfg.train.learning_rate
        self.model.train()

    def forward(self, batch):
        q_cls = self.model.query_emb(**batch["queries"])
        p_cls = self.model.context_emb(**batch["passages"])
        return q_cls, p_cls

    def training_step(self, batch, batch_idx):
        q_cls, p_cls = self.forward(batch)
        
        # All gather
        if self.trainer.world_size > 1:
            gathered_p_cls = self.all_gather(p_cls, sync_grads=True)    # (G, 2B, D)
            global_p_cls = gathered_p_cls.reshape(-1, p_cls.shape[1])   # (G*2B, D)
        else:
            global_p_cls = p_cls

        score = torch.matmul(q_cls, global_p_cls.T)    # (B, G*2B), do not normalize
        
        # GPU-wise loss calculation
        local_rank = self.global_rank
        start_idx = local_rank * p_cls.shape[0]
        targets = torch.arange(start_idx, start_idx + q_cls.shape[0], device=score.device)
        
        loss = F.cross_entropy(score, targets)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=q_cls.shape[0])  # Trainer api
        return loss

    def validation_step(self, batch, batch_idx):
        q_cls, p_cls = self.forward(batch)
        
        # All gather
        if self.trainer.world_size > 1:
            gathered_p_cls = self.all_gather(p_cls, sync_grads=True)    # (G, 2B, D)
            global_p_cls = gathered_p_cls.reshape(-1, p_cls.shape[1])   # (G*2B, D)
        else:
            global_p_cls = p_cls
            
        score = torch.matmul(q_cls, global_p_cls.T)    # (B, G*2B), do not normalize
        local_rank = self.global_rank
        start_idx = local_rank * p_cls.shape[0]
        targets = torch.arange(start_idx, start_idx + q_cls.shape[0], device=score.device)
        loss = F.cross_entropy(score, targets)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=q_cls.shape[0])    # Trainer api
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)