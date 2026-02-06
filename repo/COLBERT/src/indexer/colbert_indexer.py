import faiss
import os
import pickle
import torch
import numpy as np
import torch
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Iterator
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
        
        self.index_id_to_db_id = []
        self.pid_to_int_map = {}
        
        self.codec = None
        self.ivf_index = None
        self.ndocs = cfg.ndocs
    
    def train(self, sampled_vectors: np.ndarray, num_partitions: int):
        """
        Core function of ColBERT Indexer.
        Train stage consists of 3 steps
        1. Kmeans clustering (Coarse)
        2. Residual computation (Fine-grained)
        3. Construct Codec object
        """
        logger.info(f"# of partitions: {num_partitions}")
        self.num_partitions = num_partitions
        
        sampled, heldout = self._split_vectors(sampled_vectors, 0.95)
        # Kmeans clustering
        args_ = [self.cfg.vector_dim, self.num_partitions, self.cfg.niter, sampled]
        centroids = compute_faiss_kmeans(*args_)    # (C, D)
        c_norm = torch.norm(centroids, dim=1, keepdim=True) + 1e-10
        centroids = (centroids / c_norm)
        
        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)
        self.codec = ResidualCodec(
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights
        )

    def _split_vectors(self, vectors: np.ndarray, sample_prob: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        split_idx = int(vectors.shape[0] * sample_prob)
        sampled = vectors[:split_idx]
        heldout = vectors[split_idx:]
        return sampled, heldout
    
    def _compute_avg_residual(self, centroids: torch.Tensor, heldout: np.ndarray):
        device = centroids.device
        
        # Safe normalize
        heldout = torch.from_numpy(heldout).to(device)
        heldout = torch.nn.functional.normalize(heldout, p=2, dim=1)
        
        bsize = self.cfg.index_bsize
        heldout_size = heldout.shape[0]
        
        # Search & compute residuals
        c_indices = []
        for start_idx in range(0, heldout_size, bsize):
            end_idx = min(heldout_size, start_idx + bsize)
            batch_queries = heldout[start_idx:end_idx]   # (B, D)
            scores = batch_queries @ centroids.T   # (B, C)
            I = torch.argmax(scores, dim=1)   # (B,)
            c_indices.append(I)
        c_indices = torch.cat(c_indices, dim=0)   # (N,)
        
        nearest_centroids = centroids[c_indices]
        residuals = heldout - nearest_centroids
        avg_residual = torch.mean(torch.abs(residuals), dim=0)  # (D,), scale
        
        normalized_residuals = residuals / (avg_residual + 1e-10)

        # Quantization information
        num_options = 2 ** self.cfg.nbits
        flat_residuals = normalized_residuals.view(-1)
        quantiles_p = torch.linspace(0, 1, steps=num_options + 1, device=device)[1:-1]  # (Q-1,)
        bucket_cutoffs = torch.quantile(flat_residuals, quantiles_p)
        
        weights_p = torch.linspace(0, 1, steps=2*num_options + 1, device=device)[1::2]   # for reconstruction
        bucket_weights = torch.quantile(flat_residuals, weights_p)
        
        return (
            bucket_cutoffs.cpu(),
            bucket_weights.cpu(),
            avg_residual.cpu()
        )
    
    def index_data(self, emb_iterator: Iterator, rank: int):
        """
        Index data from the embeds iterator which is provided by IndexFileManager
        """
        for batch_vectors, batch_pids in emb_iterator:
            batch_vectors = batch_vectors.to(rank)
            batch_vectors = torch.nn.functional.normalize(batch_vectors, p=2, dim=1)
            codes, indices = self.codec.compress(batch_vectors)
            self.buffer["codes"].append(codes)
            self.buffer["indices"].append(indices)
            
            batch_pids_int = []
            if isinstance(batch_pids, np.ndarray):
                batch_pids = batch_pids.tolist()
            
            for pid in batch_pids:
                if pid not in self.pid_to_int_map:
                    new_id = len(self.index_id_to_db_id)
                    self.index_id_to_db_id.append(pid)
                    self.pid_to_int_map[pid] = new_id
                batch_pids_int.append(self.pid_to_int_map[pid])
            self.buffer['pids'].extend(batch_pids_int)
            
            current_buffer_size = sum(c.shape[0] for c in self.buffer['codes'])
            if current_buffer_size >= self.chunk_size_limit:
                self._flush_buffer()
        if len(self.buffer["codes"]) > 0:
            self._flush_buffer()
            
    def build_ivf(self):
        """
        Build IVF index from the indexed data
        """
        logger.info("Building IVF index")
        codes_list, indices_list, pids_list = [], [], []
        
        for chunk_idx in range(self.chunk_idx):
            data = torch.load(f"{self.cfg.output_dir}/{chunk_idx}.pt")
            codes_list.append(data["codes"])
            indices_list.append(data["indices"])
            pids_list.append(data["pids"])
        
        all_codes = torch.cat(codes_list, dim=0)  # (Total_N, D_packed)
        all_indices = torch.cat(indices_list, dim=0)  # (Total_N,)
        all_pids = torch.cat(pids_list, dim=0)  # (N,)
        logger.info(f"Total vectors: {len(all_indices)}")
        
        # sort by centroid order
        sort_order = torch.argsort(all_indices)
        ivf_codes = all_codes[sort_order]
        ivf_pids = all_pids[sort_order]
        
        ivf_lengths = torch.bincount(all_indices[sort_order], minlength=self.num_partitions)
        ivf_indptr = torch.cat([torch.tensor([0]), torch.cumsum(ivf_lengths, dim=0)])  # Similar to CSR sparse matrix pointer

        self.ivf_index = {
            "codes": ivf_codes,
            "pids": ivf_pids,
            "indptr": ivf_indptr
        }
        logger.info("Complete IVF index building.")
    
    def _flush_buffer(self):
        codes_concat = torch.vstack(self.buffer["codes"]) # (N, D_packed)
        indices_concat = torch.cat(self.buffer["indices"], dim=0) # (N,)
        pids_concat = torch.tensor(self.buffer["pids"], dtype=torch.long)
        
        save_path = os.path.join(self.cfg.output_dir, f"{self.chunk_idx}.pt")
        data_to_save = {
            "codes": codes_concat,      # Tensor (uint8)
            "indices": indices_concat,  # Tensor (int32)
            "pids": pids_concat,        # Tensor (long)
        }

        torch.save(data_to_save, save_path)
        logger.info(f"Saved chunk {self.chunk_idx} to {save_path} ({len(pids_concat)} vectors)")
        
        self.buffer = {"codes": [], "indices": [], "pids": []}
        self.chunk_idx += 1
    
    def search(self, query_vectors: torch.Tensor, query_indices: List[str], top_k: int, batch_size: int = 128, nprobe: int = 4) -> Dict[str, List[str]]:
        logger.info(f"Searching for top {top_k} nearest neighbors...")
        
        n = query_vectors.shape[0]
        iterator = tqdm(range(0, n, batch_size), desc="Searching queries")

        total_indices = []
        avg_hits = []
        for start_idx in iterator:
            end_idx = min(n, start_idx+batch_size)
            batch_queries = query_vectors[start_idx:end_idx]    # (B, L_q, D)
            
            # TODO: Search logic
            scores, indices, avg_hit_per_query_token = self.codec.search(self.ivf_index, batch_queries, top_k, nprobe, self.ndocs)
            total_indices.append(indices)
            avg_hits.append(avg_hit_per_query_token)
        total_indices = np.vstack(total_indices)
        avg_hit = sum(avg_hits) / len(avg_hits)
        logger.info(f"Average hits per query token: {avg_hit:.2f}")
        
        # mapping faiss internal ids to db ids
        mapped_ids = {}
        for query_idx, topk_indices in zip(query_indices, total_indices):
            topk_mapped = [self.index_id_to_db_id[idx] if idx != -1 else None for idx in topk_indices]
            mapped_ids[query_idx] = topk_mapped

        return mapped_ids
    
    def save(self, output_dir: str):
        # Save metadata
        save_path = os.path.join(output_dir, "index_meta.pkl")
        save_to_disk = {
            "index_id_to_db_id": self.index_id_to_db_id,
            "pid_to_int_map": self.pid_to_int_map
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_to_disk, f)
        logger.info(f"Save index metadata to {save_path}.")
        
        # Save IVF index
        save_path = os.path.join(self.cfg.output_dir, "ivf_index.pt")
        torch.save(self.ivf_index, save_path)
        logger.info(f"Save IVF index to {save_path}.")
        
        # Save ResidualCodec
        codec_dir = os.path.join(output_dir, "codec")
        self.codec.save(codec_dir)
        logger.info(f"Save ResidualCodec to {codec_dir}.")
    
    def load(self, index_dir: str):
        # Load metadata
        meta_path = os.path.join(index_dir, "index_meta.pkl")
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        self.index_id_to_db_id = metadata["index_id_to_db_id"]
        self.pid_to_int_map = metadata["pid_to_int_map"]
        logger.info(f"Load metadata from {meta_path}.")
        
        # Load IVF index
        self.ivf_index = torch.load(f"{index_dir}/ivf_index.pt")
        logger.info(f"IVF index loaded from {index_dir}/ivf_index.pt.")
        
        # Load ResidualCodec
        codec_dir = f"{index_dir}/codec"
        self.codec = ResidualCodec.load(codec_dir)
        logger.info(f"Codec loaded from {codec_dir}.")


# Reference: https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/indexing/collection_indexer.py#L500
def compute_faiss_kmeans(
    dim: int,
    num_partitions: int,
    kmeans_niters: int,
    sampled_vectors: np.ndarray,
    use_gpu: bool = True
) -> torch.Tensor:
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=use_gpu, verbose=True, seed=42)
    kmeans.train(sampled_vectors)
    centroids = torch.from_numpy(kmeans.centroids)
    return centroids