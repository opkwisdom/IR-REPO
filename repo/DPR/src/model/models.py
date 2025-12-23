from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.cfg.model_name_or_path)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token representation

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)



class DPREncoder(BiEncoder, nn.Module):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.question_model = Encoder(cfg)
        self.context_model = Encoder(cfg)

    def query_emb(self, input_ids, attention_mask):
        return self.question_model.encode(input_ids, attention_mask)

    def context_emb(self, input_ids, attention_mask):
        return self.context_model.encode(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask):
        query_embeddings = self.query_emb(input_ids, attention_mask)
        context_embeddings = self.context_emb(input_ids, attention_mask)
        return query_embeddings, context_embeddings
