from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token representation


class DPREncoder(BiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.query_model = Encoder(cfg.model_name_or_path)
        self.context_model = Encoder(cfg.model_name_or_path)

    def query_emb(self, input_ids, attention_mask):
        return self.query_model(input_ids, attention_mask)

    def context_emb(self, input_ids, attention_mask):
        return self.context_model(input_ids, attention_mask)

    def forward(self, q_inputs, p_inputs):
        q_cls = self.query_model(**q_inputs)    # (B, D)
        p_cls = self.context_model(**p_inputs)  # (2*B, D), (pos + neg)
        return (q_cls, p_cls)
