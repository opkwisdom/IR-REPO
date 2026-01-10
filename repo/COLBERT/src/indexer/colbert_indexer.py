import faiss
import os
import pickle
import torch
import numpy as np
import torch
from typing import List, Dict
from omegaconf import DictConfig


class ColBERTIndexer:
    """
    A custom indexer which utilizes compression-based approach (IVF + RQ),
    and MaxSim-based search operation.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.index_type = index_type
        self.index_id_to_db_id = []
    
    def _setup_for_train(self):
        pass
    
    def train(self):
        pass
    
    def search(self, query_vectors: torch.Tensor, query_indices: List[str], top_k: int, batch_size: int = 128) -> Dict[str, List[str]]:
        pass
    
    def load(self):
        pass
    
    def save(self):
        pass