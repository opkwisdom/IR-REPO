from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, model_name_or_path: str, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def forward(self, input_ids, attention_mask):
        CHUNK_SIZE = 8  # To prevent OOM
        total_size = input_ids.shape[0]

        sparse_emb = []
        for start_idx in range(0, total_size, CHUNK_SIZE):
            end_idx = min(total_size, start_idx + CHUNK_SIZE)
            sub_input_ids = input_ids[start_idx:end_idx, :]
            sub_attention_mask = attention_mask[start_idx:end_idx, :]
            outputs = self.model(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
            logits = outputs.logits     # (sub, L, V), All token representation
            sparse_rep = torch.log(1 + F.relu(logits))
            sparse_rep = sparse_rep * sub_attention_mask.unsqueeze(-1)
            sub_sparse_emb = torch.amax(sparse_rep, dim=1)
            sparse_emb.append(sub_sparse_emb)
        sparse_emb = torch.vstack(sparse_emb)   # (B, V)
        return sparse_emb


class SpladeEncoder(BiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.share_encoder:
            self.query_model = Encoder(cfg.model_name_or_path, cfg.use_gradient_checkpointing)
            self.context_model = self.query_model
        else:
            self.query_model = Encoder(cfg.model_name_or_path, cfg.use_gradient_checkpointing)
            self.context_model = Encoder(cfg.model_name_or_path, cfg.use_gradient_checkpointing)

    def query_emb(self, input_ids, attention_mask):
        return self.query_model(input_ids, attention_mask)

    def context_emb(self, input_ids, attention_mask):
        return self.context_model(input_ids, attention_mask)

    def forward(self, q_inputs, p_inputs):
        q_sparse_emb = self.query_model(**q_inputs)    # (B, V)
        p_sparse_emb = self.context_model(**p_inputs)  # (2*B, V), (pos + neg)
        return (q_sparse_emb, p_sparse_emb)