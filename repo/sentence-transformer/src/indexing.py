import os
import pickle
import torch
import hydra
import logging
import glob
import numpy as np
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union, List, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from indexer import FaissIndexer
from utils import (
    set_seed,
    load_collection,
    get_best_checkpoint,
    setup, cleanup,
)

logger = logging.getLogger(__file__)

def gen_ctx_vectors(
    rank: int,
    shard_collection: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    cfg: DictConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate ctx vectors, each GPU process
    """
    device = torch.device(f"cuda:{rank}")

    all_pids = list(shard_collection.keys())
    all_texts = [shard_collection[pid] for pid in all_pids]
    total_len = len(all_texts)
    logger.info(f"Generating context vectors for {total_len} passages...")
    
    batch_size = cfg.search.batch_size
    ctx_vectors = []
    ctx_ids = []

    iterator = tqdm(range(0, total_len, batch_size), desc=f"GPU {rank}", position=rank)
    for start_idx in iterator:
        end_idx = min(total_len, start_idx + batch_size)
        batch_ids = all_pids[start_idx:end_idx]
        batch_data = all_texts[start_idx:end_idx]

        batch_titles = [d["title"] for d in batch_data]
        batch_texts = [d["contents"] for d in batch_data]

        batch_inputs = [
            title + " " + text
            for title, text in zip(batch_titles, batch_texts)
        ]
        with torch.no_grad():
            ctx_embs = encoder.encode(
                batch_inputs,
                show_progress_bar=False,
                batch_size=batch_size,
                device=device
            )   # (B, D)
        ctx_vectors.append(ctx_embs)
        ctx_ids.extend(batch_ids)

    ctx_vectors = np.vstack(ctx_vectors)  # (N, D)
    return ctx_vectors, ctx_ids


def multi_gpu_worker(rank: int, world_size: int, collection_shards: List[Dict[str, Dict[str, str]]], cfg: DictConfig):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    # Load SentenceTransformer model
    try:
        logger.info(f"[GPU {rank}] Loading model...")
        context_encoder = SentenceTransformer(cfg.model.model_name_or_path)
        device = torch.device(f"cuda:{rank}")
        context_encoder.to(device)
        context_encoder.eval()

        vectors, ids = gen_ctx_vectors(rank, collection_shards[rank], context_encoder, cfg)
        tmp_path = os.path.join(cfg.output_dir, f"shard_{rank}.pkl")
        with open(tmp_path, 'wb') as f:
            pickle.dump((vectors, ids), f)
    finally:
        cleanup()


def get_shards(collection: Dict[str, Dict[str, str]], num_shards: int) -> List[Dict[str, Dict[str, str]]]:
    all_pids = list(collection.keys())
    shard_size = len(all_pids) // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        # 마지막 샤드는 남은 데이터를 모두 포함
        end = (i + 1) * shard_size if i < num_shards - 1 else len(all_pids)
        shard_pids = all_pids[start:end]
        shards.append({pid: collection[pid] for pid in shard_pids})
    return shards


@hydra.main(version_base=None, config_path="../conf", config_name="indexing")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    world_size = torch.cuda.device_count()

    # Load collection
    collection = load_collection(cfg.dataset.collection_path, logger)
    shards = get_shards(collection, world_size)
    # shards = get_shards(collection, 1)  # For debugging without multi-gpu

    # Multi-Process DDP indexing
    os.makedirs(cfg.output_dir, exist_ok=True)
    mp.spawn(multi_gpu_worker, args=(world_size, shards, cfg), nprocs=world_size, join=True)
    # multi_gpu_worker(0, 1, shards, cfg)  # For debugging without multi-gpu

    # Gather all vectors and ids from shards
    ctx_vectors = []
    ctx_ids = []
    logger.info("Gathering context vectors from all shards...")
    for i in range(world_size):
        tmp_path = os.path.join(cfg.output_dir, f"shard_{i}.pkl")
        with open(tmp_path, 'rb') as f:
            shard_vectors, shard_ids = pickle.load(f)
            ctx_vectors.append(shard_vectors)
            ctx_ids.extend(shard_ids)
        os.remove(tmp_path)
    ctx_vectors = np.vstack(ctx_vectors)  # (N, D)
    logger.info(f"Total context vectors shape: {ctx_vectors.shape}")

    # Build FAISS index
    cfg_index = cfg.index[cfg.index_key]
    indexer = FaissIndexer(cfg_index)
    indexer.index_data(ctx_ids, ctx_vectors, buffer_size=cfg_index.buffer_size)
    output_dir = os.path.join(cfg.output_dir, cfg.model.model_name_or_path.replace('/', '_'))
    os.makedirs(output_dir, exist_ok=True)
    indexer.save(os.path.join(output_dir, f"{cfg.dataset.name}_faiss"))

if __name__ == "__main__":
    main()