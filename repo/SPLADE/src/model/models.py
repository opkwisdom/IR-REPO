from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V), All token representation
        sparse_rep = torch.log(1 + F.relu(logits))
        sparse_rep = sparse_rep * attention_mask.unsqueeze(-1)  # apply padding
        sparse_emb = torch.amax(sparse_rep, dim=1)   # (B, V)
        return sparse_emb


class SpladeEncoder(BiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.query_model = Encoder(cfg.model_name_or_path)
        self.context_model = Encoder(cfg.model_name_or_path)

    def query_emb(self, input_ids, attention_mask):
        return self.query_model(input_ids, attention_mask)

    def context_emb(self, input_ids, attention_mask):
        return self.context_model(input_ids, attention_mask)

    def forward(self, q_inputs, p_inputs):
        q_sparse_emb = self.query_model(**q_inputs)    # (B, V)
        p_sparse_emb = self.context_model(**p_inputs)  # (2*B, V), (pos + neg)
        return (q_sparse_emb, p_sparse_emb)
