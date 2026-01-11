import os
import re
import glob
import pickle
import logging
import numpy as np
from typing import Tuple, List, Iterator

logger = logging.getLogger()

class IndexFileManager:
    """
    Manages the loading and global offset mapping of large-scale vector index partitions
    """
    def __init__(self, index_dir: str):
        self.npy_files = sorted(
            glob.glob(f"{index_dir}/*.npy"),
            key = lambda x: self._parse_rank_chunk(x)
        )
        self.index_dir = index_dir
        # global_offsets and file map are metadata
        self.global_offsets = [0]
        self.file_map = []
        cur_offset = 0

        for fpath in self.npy_files:
            count = np.load(fpath, mmap_mode='r').shape[0]
            self.file_map.append({
                "path": fpath,
                "count": count,
                "start_idx": cur_offset
            })
            cur_offset += count
            self.global_offsets.append(cur_offset)
        
        self.total_count = cur_offset
        logger.info(f"Total vectors: {self.total_count}")
        
    def _parse_rank_chunk(self, filepath):
        """
        something/shard_0_50.npy -> (0, 50)
        """
        fname = os.path.basename(filepath)
        match = re.search(r'shard_(\d+)_(\d+).npy', fname)
        if match:
            return int(match.group(1)), int(match.group(2))
    
    def __len__(self):
        return self.total_count
    
    def sample_vectors(self, k: int, seed: int = 42) -> Tuple[np.ndarray, int]:
        """
        Sample embeddings for Kmeans clustering
        """
        np.random.seed(seed)
        vectors_est = k * np.sqrt(self.total_count)
        vectors_est = min(1 + int(vectors_est), self.total_count)
        num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(self.total_count))))    # from colbert codebase, collection_indexer.py:106
        
        sampled_global_indices = np.sort(np.random.choice(self.total_count, vectors_est, replace=False))
        total_sampled_vectors = []
        
        cur_idx_ptr = 0
        num_indices = len(sampled_global_indices)
        
        for meta in self.file_map:
            if cur_idx_ptr >= num_indices:
                break
            
            start_idx = meta["start_idx"]
            end_idx = start_idx + meta["count"]
            
            batch_global_indices = []
            while cur_idx_ptr < num_indices:
                g_idx = sampled_global_indices[cur_idx_ptr]
                if g_idx >= end_idx:
                    break
                
                batch_global_indices.append(g_idx)
                cur_idx_ptr += 1
            
            if batch_global_indices:
                local_indices = [idx - start_idx for idx in batch_global_indices]
                shard_data = np.load(meta['path'], mmap_mode='r')
                vectors = shard_data[local_indices]     # (B, D)
                total_sampled_vectors.append(vectors)
        
        return np.vstack(total_sampled_vectors), num_partitions
        
    def stream_batches(self, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields:
            (vectors, pids)
        """
        for meta in self.file_map:
            chunk_vecs = np.load(meta['path'], mmap_mode='r')
            with open(meta['path'].replace('.npy', '.pkl'), 'rb') as f:
                chunk_metadata = pickle.load(f)
            chunk_pids = chunk_metadata["pids"]
            chunk_doclens = chunk_metadata["doclens"]
            
            pids_arr = np.array(chunk_pids)
            expanded_pids = np.repeat(pids_arr, chunk_doclens)
            
            total_vectors = chunk_vecs.shape[0]
            if len(expanded_pids) != total_vectors:
                logger.error(f"Length mismatch: {len(expanded_pids)}, {total_vectors}")
            
            for start_idx in range(0, total_vectors, batch_size):
                end_idx = min(total_vectors, start_idx + batch_size)
                yield chunk_vecs[start_idx:end_idx], expanded_pids[start_idx:end_idx]
    
    def finalize(self):
        """
        Clean-up intermediate files
        """
        inter_files = glob.glob(f"{self.index_dir}/shard_*")
        for fpath in inter_files:
            os.remove(fpath)