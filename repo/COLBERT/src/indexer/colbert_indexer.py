import faiss
import os
import pickle
import torch
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple
from omegaconf import DictConfig

from .residual_codec import ResidualCodec

logger = logging.getLogger()

class ColBERTIndexer:
    """
    A custom indexer which utilizes compression-based approach (IVF + RQ),
    and MaxSim-based search operation.
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.index_type = cfg.index_type
        self.buffer = {"codes": [], "indices": [], "pids": []}
        self.chunk_idx = 0
        self.chunk_size_limit = 1_000_000
        self.codec = None
    
    def train(self, sampled_vectors: np.ndarray, num_partitions: int, codec_dir: str):
        """
        Core function of ColBERT Indexer.
        Train stage consists of 3 steps
        1. Kmeans clustering (Coarse)
        2. Residual computation (Fine-grained)
        3. Save Codec object
        """
        logger.info(f"# of partitions: {num_partitions}")
        self.num_partitions = num_partitions
        
        sampled, heldout = self._split_vectors(sampled_vectors, 0.95)
        # Kmeans clustering
        args_ = [self.cfg.vector_dim, self.num_partitions, self.cfg.niter, sampled]
        centroids = compute_faiss_kmeans(*args_)    # (C, D)
        c_norm = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10
        centroids = centroids / c_norm
        
        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)
        self.codec = ResidualCodec(
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights
        )
        self.codec.save(codec_dir)
    
    def load_codec(self, codec_dir: str):
        self.codec = ResidualCodec.load(codec_dir)
        logger.info("Codec loaded from disk.")

    def _split_vectors(self, vectors: np.ndarray, sample_prob: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        split_idx = int(vectors.shape[0] * sample_prob)
        sampled = vectors[:split_idx]
        heldout = vectors[split_idx:]
        return sampled, heldout
    
    def _compute_avg_residual(self, centroids: np.ndarray, heldout: np.ndarray):
        # Safe normalize
        h_norm = np.linalg.norm(heldout, axis=1, keepdims=True) + 1e-10
        heldout = heldout / h_norm
        
        # compute residual using faiss FlatIP
        dim = centroids.shape[1]
        flat_index = faiss.IndexFlatIP(dim)
        
        if self.cfg.use_gpu:
            res = faiss.StandardGpuResources()
            flat_index = faiss.index_cpu_to_gpu(res, 0, flat_index)
        
        flat_index.add(centroids)
        bsize = self.cfg.index_bsize
        heldout_size = heldout.shape[0]
        
        # Search & compute residuals
        c_indices = []
        for start_idx in range(0, heldout_size, bsize):
            end_idx = min(heldout_size, start_idx + bsize)
            batch_queries = heldout[start_idx:end_idx]
            _, I = flat_index.search(batch_queries, 1)    # (B, 1)
            c_indices.extend(I.squeeze(1))
        
        nearest_centroids = centroids[c_indices]
        residuals = heldout - nearest_centroids
        avg_residual = np.mean(np.abs(residuals), axis=0)   # (D,), scale
        
        # Quantization information
        num_options = 2 ** self.cfg.nbits
        quantiles = np.arange(0, num_options) * (1 / num_options)
        bucket_cutoffs = np.quantile(residuals, quantiles[1:])   # (Q,), global cutoff
        
        w_quantiles = quantiles + (0.5 / num_options)   # for reconstruction
        bucket_weights = np.quantile(residuals, w_quantiles)
        
        return bucket_cutoffs, bucket_weights, avg_residual
    
    def index_data(self, emb_iterator):
        """
        Index data from the embeds iterator which is provided by IndexFileManager
        """
        for batch_vectors, batch_pids in emb_iterator:
            codes, indices = self.codec.compress(batch_vectors)
            self.buffer["codes"].append(codes)
            self.buffer["indices"].append(indices)
            
            if isinstance(batch_pids, np.ndarray):
                self.buffer['pids'].extend(batch_pids.tolist())
            else:
                self.buffer['pids'].extend(batch_pids)
            
            current_buffer_size = sum(c.shape[0] for c in self.buffer['codes'])
            if current_buffer_size >= self.chunk_size_limit:
                self._flush_buffer()
        if len(self.buffer["codes"]) > 0:
            self._flush_buffer()
    
    def _flush_buffer(self):
        codes_concat = np.concatenate(self.buffer["codes"], axis=0) # (N, D_packed)
        indices_concat = np.concatenate(self.buffer["indices"], axis=0) # (N,)
        pids_concat = np.array(self.buffer["pids"])
        
        save_path = os.path.join(self.cfg.output_dir, f"{self.chunk_idx}.npz")
        np.savez_compressed(
            save_path,
            codes=codes_concat,
            indices=indices_concat,
            pids=pids_concat
        )
        logger.info(f"Saved chunk {self.chunk_idx} to {save_path} ({len(pids_concat)} vectors)")
        
        self.buffer = {"codes": [], "indices": [], "pids": []}
        self.chunk_idx += 1
    
    def search(self, query_vectors: torch.Tensor, query_indices: List[str], top_k: int, batch_size: int = 128) -> Dict[str, List[str]]:
        pass
    
    def load(self, path_prefix: str):
        pass
    
    def save(self, path_prefix: str):
        pass


# Reference: https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/indexing/collection_indexer.py#L500
def compute_faiss_kmeans(
    dim: int,
    num_partitions: int,
    kmeans_niters: int,
    sampled_vectors: np.ndarray,
    use_gpu: bool = True
) -> np.ndarray:
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=42)
    kmeans.train(sampled_vectors)
    
    return kmeans.centroids