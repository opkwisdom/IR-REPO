import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from omegaconf import DictConfig
from typing import List, Dict
from transformers import get_linear_schedule_with_warmup

from .models import DPREncoder
from utils import evaluate_search_results


class DPRLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: DPREncoder):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = cfg.train.learning_rate
        self.model.train()
        self.val_results: Dict[str, List[str]] = {}

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
        batch_dict = {"queries": batch["queries"], "passages": batch["passages"]}
        q_cls, p_cls = self.forward(batch_dict)
        score = torch.matmul(q_cls, p_cls.T).squeeze(0)  # (Top-k,)
        sorted_indices = torch.argsort(score, descending=True)
        
        passage_ids = batch["passage_ids"]
        sorted_passage_ids = [passage_ids[i] for i in sorted_indices]

        q_id = batch["query_id"]
        self.val_results[q_id] = sorted_passage_ids

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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

