# This is an interface definition for a Bi-Encoder model in a DPR system.
from abc import ABC, abstractmethod
import torch.nn as nn

class BiEncoder(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def query_emb(self, input_ids, attention_mask):
        """Compute query embeddings."""
        pass    

    @abstractmethod
    def context_emb(self, input_ids, attention_mask):
        """Compute context embeddings."""
        pass

    @abstractmethod
    def forward(self, input_ids, attention_mask):
        """Forward pass for the Bi-Encoder."""
        pass