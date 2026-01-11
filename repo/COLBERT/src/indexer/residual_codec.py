import os
import logging
import numpy as np

logger = logging.getLogger()

class ResidualCodec:
    def __init__(
        self,
        centroids: np.ndarray,
        avg_residual: np.ndarray,
        bucket_cutoffs: np.ndarray,
        bucket_weights: np.ndarray
    ):
        self.centroids = centroids
        self.avg_residual = avg_residual
        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights
        self.nbits = int(np.log2(len(self.bucket_weights)))
        
    def compress(self, batch_vectors: np.ndarray):
        """
        Compress batch vectors into codes
        Args:
            batch_vectors: (B, D) vectors
        Returns:
            packed_codes: (B, D_packed) uint8
            centroid_indices: (B,) int32
        """
        I = np.argmax(batch_vectors @ self.centroids.T, axis=1) # (B,)
        nearest_centroids = self.centroids[I]
        residuals = batch_vectors - nearest_centroids    # (B, D)
        raw_codes = self.compress_into_codes(residuals, self.bucket_cutoffs)    # (B, D)

        # bit packing
        packed_codes = self.pack_codes(raw_codes)
        centroid_indices = I.astype(np.int32)
        return packed_codes, centroid_indices
    
    def compress_into_codes(self, residuals, bucket_cutoffs):
        """
        Args:
            residuals: (N, D) vectors
            bucket_cutoffs: (Q-1,)
        Returns:
            codes: (N, D)
        """
        codes = np.sum(residuals[:,:,None] > bucket_cutoffs, axis=-1)
        return codes
    
    def pack_codes(self, raw_codes: np.ndarray):
        """
        Args:
            codes: (B, D) int array
        Returns:
            packed_codes: (B, D_packed) uint8 array
        """
        packing_factor = 8 // self.nbits
        B, D = raw_codes.shape
        packed = np.zeros((B, D // packing_factor), dtype=np.uint8)
        
        for i in range(packing_factor):
            cur_cols = raw_codes[:, i::packing_factor]
            shift_amount = 8 - (self.nbits * (i+1))
            packed |= (cur_cols.astype(np.uint8) << shift_amount)   # OR operation
        return packed
    
    def decompress(self, packed_codes: np.ndarray, indices: np.ndarray):
        pass
    
    def save(self, output_dir: str):
        logger.info(f"Save all codec objects to {output_dir}.")
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/centroids.npy", self.centroids)
        np.save(f"{output_dir}/avg_residual.npy", self.avg_residual)
        np.save(f"{output_dir}/bucket_cutoffs.npy", self.bucket_cutoffs)
        np.save(f"{output_dir}/bucket_weights.npy", self.bucket_weights)
        logger.info("Save complete.")
    
    @classmethod
    def load(self, output_dir: str):
        logger.info(f"Load all codec objects from {output_dir}.")
        centroids = np.load(f"{output_dir}/centroids.npy")
        avg_residual = np.load(f"{output_dir}/avg_residual.npy")
        bucket_cutoffs = np.load(f"{output_dir}/bucket_cutoffs.npy")
        bucket_weights = np.load(f"{output_dir}/bucket_weights.npy")
        
        return ResidualCodec(
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights
        )