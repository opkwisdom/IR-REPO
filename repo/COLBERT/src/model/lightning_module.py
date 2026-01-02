import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Dict, List
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup

from .models import ColBERTEncoder
from utils import evaluate_search_results


class ColBERTLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: ColBERTEncoder):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg.train
        self.learning_rate = cfg.train.learning_rate
        self.model.train()
        self.val_results: Dict[str, List[str]] = {}

    def forward(self, batch):
        q_emb = self.model.query_emb(**batch["queries"])
        p_emb = self.model.context_emb(**batch["passages"])
        return q_emb, p_emb
    
    def training_step(self, batch, batch_idx):
        q_emb, p_emb = self.forward(batch)

    def validation_step(self, batch, batch_idx):
        return
    
    def on_validation_epoch_end(self):
        val_loader = self.trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]
        dev_qrels = val_loader.dataset.get_qrels()

        eval_results = evaluate_search_results(
            self.val_results,
            dev_qrels,
            k_values=[10]
        )
        self.log("val_mrr_10", eval_results["MRR@10"], sync_dist=True)
        self.log("val_recall_10", eval_results["Recall@10"], sync_dist=True)
        self.log("val_ndcg_10", eval_results["nDCG@10"], sync_dist=True)
        self.val_results.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.trainer.max_steps == -1:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_steps
        warmup_steps = self.hparams.cfg.train.warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.cfg.gradient_accumulation_steps,
            }
        }