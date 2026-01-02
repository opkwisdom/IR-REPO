from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch

from .interface import BiEncoder


class Encoder(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.1, pooling_type: str = "cls"):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.pooling_type = pooling_type
    
    def pool(self, hidden_states, attention_mask):
        if self.pooling_type == "cls":
            return hidden_states[:, 0, :]  # CLS token
        elif self.pooling_type == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool(outputs.last_hidden_state, attention_mask)


class DPREncoder(BiEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.share_encoder:
            self.query_model = Encoder(cfg.model_name_or_path, pooling_type=cfg.pooling_type)
            self.context_model = self.query_model
        else:
            self.query_model = Encoder(cfg.model_name_or_path, pooling_type=cfg.pooling_type)
            self.context_model = Encoder(cfg.model_name_or_path, pooling_type=cfg.pooling_type)

    def query_emb(self, input_ids, attention_mask):
        return self.query_model(input_ids, attention_mask)

    def context_emb(self, input_ids, attention_mask):
        return self.context_model(input_ids, attention_mask)

    def forward(self, q_inputs, p_inputs):
        q_cls = self.query_model(**q_inputs)    # (B, D)
        p_cls = self.context_model(**p_inputs)  # (2*B, D), (pos + neg)
        return (q_cls, p_cls)
