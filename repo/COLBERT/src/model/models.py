import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, model_name_or_path: str, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state    # (B, L, D), all token representation


class ColBERTEncoder(BiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        # ColBERT use single encoder
        self.query_model = Encoder(cfg.model_name_or_path)
        self.context_model = self.query_model

    def query_emb(self, input_ids, attention_mask):
        return self.query_model(input_ids, attention_mask)
    
    def context_emb(self, input_ids, attention_mask):
        return self.context_model(input_ids, attention_mask)
    
    def forward(self, q_inputs, p_inputs):
        q_emb = self.query_model(**q_inputs)    # (B, L_q, D)
        p_emb = self.context_model(**p_inputs)  # (2*B, L_p, D), (pos + neg)
        return (q_emb, p_emb)