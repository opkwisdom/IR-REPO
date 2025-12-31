import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Dict, List
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup

from .models import SpladeEncoder
from utils import evaluate_search_results


class SpladeLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, model: SpladeEncoder):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.cfg = cfg.train
        self.learning_rate = cfg.train.learning_rate
        self.model.train()
        self.val_results: Dict[str, List[str]] = {}
        
        # Set lambda (scheduled)
        self.lambda_q = self.cfg.FLOPS.lambda_q
        self.lambda_p = self.cfg.FLOPS.lambda_p

    def forward(self, batch):
        q_sparse_emb = self.model.query_emb(**batch["queries"])
        p_sparse_emb = self.model.context_emb(**batch["passages"])
        return q_sparse_emb, p_sparse_emb

    def training_step(self, batch, batch_idx):
        q_sparse_emb, p_sparse_emb = self.forward(batch)
        
        # All gather
        if self.trainer.world_size > 1:
            gathered_p_sparse_emb = self.all_gather(p_sparse_emb, sync_grads=True)    # (G, 2B, V)
            global_p_sparse_emb = gathered_p_sparse_emb.reshape(-1, p_sparse_emb.shape[1])   # (G*2B, V)
        else:
            global_p_sparse_emb = p_sparse_emb

        score = torch.matmul(q_sparse_emb, global_p_sparse_emb.T)    # (B, G*2B), do not normalize
        
        # GPU-wise loss calculation
        local_rank = self.global_rank
        start_idx = local_rank * p_sparse_emb.shape[0]
        targets = torch.arange(start_idx, start_idx + q_sparse_emb.shape[0], device=score.device)
        
        # rank loss, regularization loss
        rank_loss = F.cross_entropy(score, targets)
        mean_q_act = torch.mean(q_sparse_emb, dim=0)    # (V,)
        mean_p_act = torch.mean(p_sparse_emb, dim=0)    # (V,)
        reg_q_loss = torch.sum(mean_q_act ** 2)
        reg_p_loss = torch.sum(mean_p_act ** 2)

        self.compute_lambdas()  # Lambda Scheduling
        loss = rank_loss + self.lambda_t_q * reg_q_loss + self.lambda_t_p * reg_p_loss

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=q_sparse_emb.shape[0])  # Trainer api
        return loss

    def validation_step(self, batch, batch_idx):
        CHUNK_SIZE = 32
        total_scores = []
        
        q_sparse_emb = self.model.query_emb(**batch["queries"])
        
        # Chunking to prevent OOM
        p_input_ids = batch["passages"]["input_ids"]
        p_attention_mask = batch["passages"]["attention_mask"]

        p_input_ids_chunks = torch.split(p_input_ids, CHUNK_SIZE, dim=0)
        p_attention_mask_chunks = torch.split(p_attention_mask, CHUNK_SIZE, dim=0)

        for input_ids, attn_mask in zip(p_input_ids_chunks, p_attention_mask_chunks):
            p_sparse_emb_chunk = self.model.context_emb(input_ids=input_ids, attention_mask=attn_mask)
            chunk_score = torch.matmul(q_sparse_emb, p_sparse_emb_chunk.T)
            total_scores.append(chunk_score)
        
        total_scores = torch.cat(total_scores, dim=1).squeeze(0)  # (Top-k,)
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

    def compute_lambdas(self):
        self.lambda_t_q = min(self.lambda_q, self.lambda_q * ((self.global_step) / (self.cfg.FLOPS.T + 1)) ** 2)
        self.lambda_t_p = min(self.lambda_p, self.lambda_p * ((self.global_step) / (self.cfg.FLOPS.T + 1)) ** 2)

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