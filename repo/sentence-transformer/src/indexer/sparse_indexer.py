import logging
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from omegaconf import DictConfig

from typing import List, Tuple, Dict
from utils import SparseVector

logger = logging.getLogger()


class SparseIndexer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.crow_indices = [0]
        self.col_indices = []
        self.values = []

        self.index_id_to_db_id = []
        self.total_docs = 0
        self.index_matrix = None
        self.vocab_size = getattr(cfg, "vocab_size", 30522)

        self.avg_n_terms = 0
        
    def index_data(self, sparse_vectors: SparseVector, top_k: int = 200):
        """
        Index the provided sparse vectors with pruning (Top-K).
        """
        self.index_id_to_db_id.extend(sparse_vectors.doc_ids)
        for tokens, weights in zip(sparse_vectors.token_ids, sparse_vectors.scores):
            # Do pruning
            paired = sorted(zip(weights, tokens), key=lambda x: x[0], reverse=True)
            if len(tokens) > top_k:
                paired = paired[:top_k]
            paired.sort(key=lambda x: x[1])
            weights, tokens = zip(*paired)
                
            self.col_indices.extend(tokens)
            self.values.extend(weights)
            
            last_ptr = self.crow_indices[-1]
            self.crow_indices.append(last_ptr + len(tokens))

            self.total_docs += 1
            self.avg_n_terms += len(tokens)
    
    def search(self, query_vectors: SparseVector, query_indices: List[str], top_k: int, batch_size: int = 128) -> Dict[str, List[str]]:
        # TODO: Implement search logic
        
        flat_token_ids = [tid for seq in query_vectors.token_ids for tid in seq]
        flat_scores = [score for seq in query_vectors.scores for score in seq]
        
        lengths = [len(seq) for seq in query_vectors.token_ids]
        crow_indices = [0]
        cur = 0
        for l in lengths:
            cur += l
            crow_indices.append(cur)
        # Convert SparseVector to csr format
        t_crow = torch.tensor(crow_indices, dtype=torch.int64)
        t_col = torch.tensor(flat_token_ids, dtype=torch.int64)
        t_values = torch.tensor(flat_scores, dtype=torch.float32)
        query_sparse_matrix = torch.sparse_csr_tensor(
            t_crow, t_col, t_values,
            size=(len(t_crow - 1), self.vocab_size)
        )
        query_dense_matrix = query_sparse_matrix.to_dense().T   # (V, N)

        n = len(t_crow - 1)
        iterator = tqdm(range(0, n, batch_size), desc="Searching queries")
        total_indices = []

        # Search using sparse matrix multiplication
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n)
            batch_queries = query_dense_matrix[:, start_idx:end_idx]    # (V, B)

            batch_score = torch.sparse.mm(self.index_matrix, batch_queries)   # (I, B)
            _, indices = batch_score.topk(dim=0, k=top_k)
            topk_indices = indices.T    # (B, Top-K)
            total_indices.append(topk_indices.cpu())
        total_indices = torch.vstack(total_indices) # (N, Top-K)
        
        # mapping internal integer ids to db ids
        mapped_ids = {}
        for query_idx, topk_indices in zip(query_indices, total_indices):
            topk_mapped = [self.index_id_to_db_id[idx] if idx != -1 else None for idx in topk_indices]
            mapped_ids[query_idx] = topk_mapped

        return mapped_ids  # (N, top_k)
    
    def save(self, path: str):
        """
        Save torch sparse matrix & meta data
        """
        meta_file = path + ".index_meta.pkl"
        index_file = path + ".index.pt"

        t_crow = torch.tensor(self.crow_indices, dtype=torch.int64)
        t_col = torch.tensor(self.col_indices, dtype=torch.int64)
        t_values = torch.tensor(self.values, dtype=torch.float32)
        sparse_matrix = torch.sparse_csr_tensor(
            t_crow, t_col, t_values,
            size=(self.total_docs, self.vocab_size)
        )
        if self.avg_n_terms > 0:
            self.avg_n_terms /= self.total_docs
        logger.info(f"Avg. token count: {self.avg_n_terms:.4f} tokens")

        torch.save(sparse_matrix, index_file)
        with open(meta_file, 'wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def load(self, path: str):
        meta_file = path + ".index_meta.pkl"
        index_file = path + ".index.pt"
        
        if not os.path.exists(meta_file) or not os.path.exists(index_file):
            raise FileNotFoundError("Index or meta data path does not exist")
        
        logger.info(f"Loading index from {index_file}...")
        sparse_matrix = torch.load(index_file, map_location='cpu')
        self.total_docs, self.vocab_size = sparse_matrix.shape
        self.index_matrix = sparse_matrix
        
        with open(meta_file, 'rb') as f:
            self.index_id_to_db_id = pickle.load(f)
        logger.info(f"Load completed. Total size: {self.index_matrix.shape}")