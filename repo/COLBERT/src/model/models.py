import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from omegaconf import DictConfig
from transformers import AutoModel

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.model = AutoModel.from_pretrained(cfg.model_name_or_path)
        self.pooler = nn.Linear(self.model.config.hidden_size, cfg.compressed_dim)
        if cfg.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        projected_output = self.pooler(outputs.last_hidden_state)  # (B, L, D), all token representation
        normalized_output = F.normalize(projected_output, p=2, dim=-1)
        if attention_mask is not None:  # mask out padding tokens
            normalized_output = normalized_output * attention_mask.unsqueeze(-1)
        return normalized_output


class ColBERTEncoder(BiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        # ColBERT use single encoder
        self.query_model = Encoder(cfg)
        self.context_model = self.query_model

    # Convinient API
    def resize_token_embeddings(self, new_vocab_size: int):
        self.query_model.model.resize_token_embeddings(new_vocab_size)

    def query_emb(self, input_ids, attention_mask):
        return self.query_model(input_ids, attention_mask)
    
    def context_emb(self, input_ids, attention_mask):
        return self.context_model(input_ids, attention_mask)
    
    def forward(self, q_inputs, p_inputs):
        q_emb = self.query_model(**q_inputs)    # (B, L_q, D)
        p_emb = self.context_model(**p_inputs)  # (2*B, L_p, D), (pos + neg)
        return (q_emb, p_emb)