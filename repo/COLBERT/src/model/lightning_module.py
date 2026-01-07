import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
from typing import Dict, List
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from .models import ColBERTEncoder
from utils import evaluate_search_results


class ColBERTLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: ColBERTEncoder, tokenizer: AutoTokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.tokenizer = tokenizer  # To access membership variable
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

        # All gather
        if self.trainer.world_size > 1:
            gathered_p_emb = self.all_gather(p_emb, sync_grads=True)
            global_p_emb = gathered_p_emb.reshape(-1, *p_emb.shape[1:])  # (G*2B, L_p, D)
        else:
            global_p_emb = gathered_p_emb

        interaction_scores = einops.einsum(
            q_emb, global_p_emb,
            "B_q L_q D, B_d L_d D -> B_q B_d L_q L_d"
        )
        batch_score = torch.sum(torch.amax(interaction_scores, dim=-1), dim=-1) # (B, G*2B)

        # GPU-wise loss calculata
        local_rank = self.global_rank
        start_idx = local_rank * p_emb.shape[0]
        targets = torch.arange(start_idx, start_idx + q_emb.shape[0], device=batch_score.device)

        loss = F.cross_entropy(batch_score, targets)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=q_emb.shape[0])  # Trainer api\
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = {"queries": batch["queries"], "passages": batch["passages"]}
        q_emb, p_emb = self.forward(batch_dict)
        interaction_scores = einops.einsum(
            q_emb, p_emb,
            "B_q L_q D, B_d L_d D -> B_q B_d L_q L_d"
        )
        total_scores = torch.sum(torch.amax(interaction_scores, dim=-1), dim=-1).squeeze(0) # (Top-K,)

        # Re-ranking
        sorted_indices = torch.argsort(total_scores, descending=True)

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
        
        # Setup scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.cfg.gradient_accumulation_steps,
            }
        }