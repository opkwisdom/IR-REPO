# This is an interface definition for a Bi-Encoder model in a DPR system.
from abc import ABC, abstractmethod


class BiEncoder(ABC):
    def __init__(self, cfg):
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
        """Forward pass for the Bi-Encoder, compute loss."""
        pass