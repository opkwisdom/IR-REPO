from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Tuple, List, Union
import hydra
import logging
import os
from transformers import AutoTokenizer
import torch
import numpy as np
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from tqdm import tqdm

from model import DPRLightningModule, Encoder
from indexer import FaissIndexer
from utils import (
    load_collection,
    set_seed,
    setup, cleanup
)

logger = logging.getLogger(__file__)

def gen_ctx_vectors(
    rank: int,
    shard_collection: Dict[str, str],
    encoder: Union[Encoder, DDP],
    tokenizer: AutoTokenizer,
    cfg: DictConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate ctx vectors, each GPU process
    """
    device = torch.device(f"cuda:{rank}")
    encoder.to(device)
    encoder.eval()

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
        batch_texts = all_texts[start_idx:end_idx]
        batch_ids = all_pids[start_idx:end_idx]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.dataset.max_passage_length,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(cfg.device)

        with torch.no_grad():
            batch_vectors = encoder(**inputs)  # (B, D)
            batch_vectors = batch_vectors.cpu().numpy()
        
        ctx_vectors.append(batch_vectors)
        ctx_ids.extend(batch_ids)
    
    ctx_vectors = np.vstack(ctx_vectors)  # (N, D)
    return ctx_vectors, ctx_ids


def ddp_worker(rank: int, world_size: int, collection_shards: List[Dict[str, str]], cfg: DictConfig):
    setup(rank, world_size)
    
    # Load Lightning model & tokenizer
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_file)
    lightning_module = DPRLightningModule.load_from_checkpoint(ckpt_path)
    context_encoder = lightning_module.model.context_model.to(rank)
    context_encoder = DDP(context_encoder, device_ids=[rank])
    context_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    vectors, ids = gen_ctx_vectors(rank, collection_shards[rank], context_encoder, tokenizer, cfg)
    tmp_path = os.path.join(cfg.output_dir, f"shard_{rank}.pkl")
    with open(tmp_path, 'wb') as f:
        pickle.dump((vectors, ids), f)

    cleanup()


def get_shards(collection: Dict[str, str], num_shards: int) -> List[Dict[str, str]]:
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

    # Multi-Process DDP indexing
    mp.spawn(ddp_worker, args=(world_size, shards, cfg), nprocs=world_size, join=True)

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
    os.makedirs(cfg.output_dir, exist_ok=True)
    indexer.save(os.path.join(cfg.output_dir, f"{cfg.dataset.name}_faiss"))

if __name__ == "__main__":
    main()