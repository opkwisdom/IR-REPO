import os
import torch
import einops
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger()

class ResidualCodec:
    def __init__(
        self,
        centroids: torch.Tensor,
        avg_residual: torch.Tensor,
        bucket_cutoffs: torch.Tensor,
        bucket_weights: torch.Tensor
    ):
        self.centroids = centroids
        self.avg_residual = avg_residual
        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights
        self.nbits = int(np.log2(len(self.bucket_weights)))
        
    def compress(self, batch_vectors: torch.Tensor):
        """
        Compress batch vectors into codes
        Args:
            batch_vectors: (B, D) vectors
        Returns:
            packed_codes: (B, D_packed) uint8
            centroid_indices: (B,) int32
        """
        I = torch.argmax(batch_vectors @ self.centroids.T, dim=1) # (B,)
        nearest_centroids = self.centroids[I]
        residuals = batch_vectors - nearest_centroids    # (B, D)
        raw_codes = self.compress_into_codes(residuals, self.bucket_cutoffs)    # (B, D)

        # bit packing
        packed_codes = self.pack_codes(raw_codes)
        centroid_indices = I.to(torch.int32)
        return packed_codes, centroid_indices
    
    def compress_into_codes(self, residuals: torch.Tensor, bucket_cutoffs: torch.Tensor):
        """
        Args:
            residuals: (N, D) vectors
            bucket_cutoffs: (Q-1,)
        Returns:
            codes: (N, D)
        """
        codes = torch.sum(residuals.unsqueeze(-1) > bucket_cutoffs, dim=-1)
        return codes
    
    def pack_codes(self, raw_codes: torch.Tensor):
        """
        Args:
            codes: (B, D) int Tensor
        Returns:
            packed_codes: (B, D_packed) uint8 Tensor
        """
        packing_factor = 8 // self.nbits
        B, D = raw_codes.shape
        packed = torch.zeros((B, D // packing_factor), dtype=torch.uint8)
        
        for i in range(packing_factor):
            cur_cols = raw_codes[:, i::packing_factor]
            shift_amount = 8 - (self.nbits * (i+1))
            packed |= (cur_cols.to(torch.uint8) << shift_amount)   # OR operation
        return packed
    
    def unpack_codes(self, packed_codes: torch.Tensor):
        """
        Args:
            packed_codes: (B, D_packed) uint8 Tensor
        Returns:
            codes: (B, D) Tensor
        """
        packing_factor = 8 // self.nbits
        B, D_comp = packed_codes.shape
        codes = torch.zeros(
            (B, D_comp * packing_factor),
            dtype=torch.long,
            device=packed_codes.device
        )
        
        # Bit operation
        mask = (1 << self.nbits) - 1
        for i in range(packing_factor):
            shift_amount = 8 - (self.nbits * (i+1))
            shifted = packed_codes >> shift_amount
            codes[:, i::packing_factor] = (shifted & mask)
        
        return codes

    def search(self, ivf_index, query_vectors: torch.Tensor, topk: int, nprobe: int, ndocs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the Top-K candidates.
        Each query token perform searching respectively, then aggregated.
        Args:
            ivf_index: The IVF index to search, (codes, pids, indptr)
            query_vectors: (B, L_q, D)
            topk: int
            nprobe: int, the number of centroids to investigate
            ndocs: int, total number of documents
        Returns:
            scores: (B, Top-k)
            indices: (B, Top-k)
        """
        ivf_pids, ivf_codes, ivf_indptr = ivf_index["pids"], ivf_index["codes"], ivf_index["indptr"]
        # Step 1. Probe centroids
        query_vectors = query_vectors.to(self.centroids.device)
        score_centroids = einops.einsum(
            query_vectors, self.centroids,
            "B L_q D, C D -> B L_q C"
        )
        score_centroids, probe_centroids = torch.topk(score_centroids, k=nprobe, dim=-1)    # (B, L_q, nprobe)
        code_lookup_table = self._precompute_table(query_vectors)   # (B, L_q, D, Q)
        B, L_q, D, Q = code_lookup_table.shape
        
        # Step 2. Gather codes (parallel gather)
        B, L_q, nprobes = probe_centroids.shape
        flat_probes = probe_centroids.view(-1)  # (N_probes,)
        
        probe_list = flat_probes.tolist()
        indptr_list = ivf_indptr.tolist()
        gather_indices_list = []
        
        for pid in probe_list:
            start = indptr_list[pid]
            end = indptr_list[pid+1]
            if start == end:
                continue
            
            rng = torch.arange(start, end, device=ivf_codes.device) # n_cand per pid
            gather_indices_list.append(rng)
        
        gather_indices = torch.cat(gather_indices_list) # (Total_cand,)
        # Parallel gather
        cand_codes = ivf_codes[gather_indices]  # (Total_cand, D_packed)
        cand_pids_cpu = ivf_pids[gather_indices.cpu()]
        cand_pids = cand_pids_cpu.to(cand_codes.device)
        
        cand_indices = self.unpack_codes(cand_codes)   # (Total_cand, D), 0, 2, 3, 1, ...
        total_cand, D = cand_indices.shape
        
        # pid mapping
        lengths = [len(x) for x in gather_indices_list]
        lengths_tensor = torch.tensor(lengths, device=query_vectors.device)
        
        # Step 3. Lookup scoring
        # (B, L_q, D, Q) -> (B, L_q, nprobe, D, Q) -> (N_probes, D, Q)
        expanded_table = code_lookup_table.unsqueeze(2)\
                            .expand(-1, -1, nprobes, -1, -1)\
                            .reshape(-1, D, Q)
        aligned_table = torch.repeat_interleave(expanded_table, lengths_tensor, dim=0)
        gather_indices = cand_indices.to(torch.long).unsqueeze(-1)  # (Total_cand, D, 1)
        gathered_scores = torch.gather(aligned_table, dim=-1, index=gather_indices)
        scores = torch.sum(gathered_scores, dim=1).squeeze(-1)  # (Total_cand,)
        
        # Step 4. Aggregation by query ids
        batch_ids_per_probe = torch.arange(B, device=scores.device).repeat_interleave(L_q * nprobe)
        batch_ids = torch.repeat_interleave(batch_ids_per_probe, lengths_tensor)
        
        final_scores = torch.zeros((B, ndocs), dtype=torch.float32, device=scores.device)
        # Instead of max-reduced, simply summing over
        final_scores.index_put_(
            (batch_ids, cand_pids),
            scores,
            accumulate=True
        )
        topk_scores, topk_indices = torch.topk(final_scores, k=topk, dim=1)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()

    def _precompute_table(self, query_vectors: torch.Tensor):
        """
        Args:
            query_vectors: (B, L_q, D)
        Returns:
            code_lookup_table: (B, L_q, D, Q)
        """
        weights = self.bucket_weights
        code_lookup_table = query_vectors.unsqueeze(-1) * weights   # Element-wise
        return code_lookup_table

    
    def save(self, output_dir: str):
        logger.info(f"Save all codec objects to {output_dir}.")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.centroids, f"{output_dir}/centroids.pt")
        torch.save(self.avg_residual, f"{output_dir}/avg_residual.pt")
        torch.save(self.bucket_cutoffs, f"{output_dir}/bucket_cutoffs.pt")
        torch.save(self.bucket_weights, f"{output_dir}/bucket_weights.pt")
        logger.info("Save complete.")
    
    @classmethod
    def load(self, output_dir: str, device: torch.device = "cuda"):
        logger.info(f"Load all codec objects from {output_dir}.")
        centroids = torch.load(f"{output_dir}/centroids.pt", map_location=device)
        avg_residual = torch.load(f"{output_dir}/avg_residual.pt", map_location=device)
        bucket_cutoffs = torch.load(f"{output_dir}/bucket_cutoffs.pt", map_location=device)
        bucket_weights = torch.load(f"{output_dir}/bucket_weights.pt", map_location=device)
        
        return ResidualCodec(
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights
        )