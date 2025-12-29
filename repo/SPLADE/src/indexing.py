import os
import pickle
import torch
import hydra
import logging
import glob
import numpy as np
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union, List
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm

from model import SpladeLightningModule, Encoder, SpladeEncoder
from indexer import SparseIndexer
from utils import (
    set_seed,
    load_collection,
    get_best_checkpoint,
    setup, cleanup
)

# {docid: {token_id: weight, ...}}
type SparseVector = Dict[str, Dict[str, float]]

logger = logging.getLogger(__file__)

def gen_ctx_sparse_vectors(
    rank: int,
    shard_collection: Dict[str, Dict[str, str]],
    encoder: Union[Encoder, DDP],
    tokenizer: AutoTokenizer,
    cfg: DictConfig
) -> SparseVector:
    """
    Generate sparse ctx vectors, each GPU process
    """
    device = torch.device(f"cuda:{rank}")
    encoder.to(device)
    encoder.eval()
    sparse_vectors: SparseVector = {}

    all_pids = list(shard_collection.keys())
    all_texts = [shard_collection[pid] for pid in all_pids]
    total_len = len(all_pids)
    batch_size = cfg.search.batch_size

    iterator = tqdm(range(0, total_len, batch_size), desc=f"GPU {rank}", position=rank)
    for start_idx in iterator:
        end_idx = min(total_len, start_idx + batch_size)
        batch_ids = all_pids[start_idx:end_idx]
        batch_data = all_texts[start_idx:end_idx]

        batch_titles = [d["title"] for d in batch_data]
        batch_texts = [d["contents"] for d in batch_data]

        inputs = tokenizer(
            text=batch_titles,
            text_pair=batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.model.max_passage_length,
            return_tensors="pt"
        ).to(device)
        import pdb; pdb.set_trace()
        with torch.no_grad():
            batch_vectors = encoder(**inputs)
        # TODO: Convert to sparse representation and store

    return sparse_vectors

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

def ddp_worker(rank: int, world_size: int, collection_shards: List[Dict[str, Dict[str, str]]], cfg: DictConfig):
    setup(rank, world_size)

    # Load Lightning model & tokenizer
    ckpt_path = get_best_checkpoint(cfg.ckpt_dir)
    backbone = SpladeEncoder(cfg.model)
    lightning_module = SpladeLightningModule.load_from_checkpoint(ckpt_path, model=backbone)
    context_encoder = lightning_module.model.context_model.to(rank)
    context_encoder = DDP(context_encoder, device_ids=[rank])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    # Generate and save chunks
    shard = collection_shards[rank]
    shard_keys = list(shard.keys())
    n = len(shard_keys)

    CHUNK_SIZE = 100000
    iterator = tqdm(range(0, n, CHUNK_SIZE), desc=f"Processing chunk on GPU-{rank}")
    for i, start_idx in enumerate(iterator):
        end_idx = min(n, start_idx + CHUNK_SIZE)
        chunk_pids = shard_keys[start_idx:end_idx]
        chunk_collection = {pid: shard[pid] for pid in chunk_pids}
        sparse_vectors = gen_ctx_sparse_vectors(rank, chunk_collection, context_encoder, tokenizer, cfg)
        tmp_path = os.path.join(cfg.output_dir, f"shard_{rank}_{i}.pkl")
        with open(tmp_path, 'wb') as f:
            pickle.dump(sparse_vectors, f)

    cleanup()

@hydra.main(version_base=None, config_path="../conf", config_name="indexing")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    world_size = torch.cuda.device_count()
    # Load collection
    collection = load_collection(cfg.data.collection_path)
    shards = get_shards(collection, world_size)

    # Multi-Process DDP indexing
    os.makedirs(cfg.output_dir, exist_ok=True)
    # mp.spawn(ddp_worker, args=(world_size, shards, cfg), nprocs=world_size)
    ddp_worker(0, world_size, shards, cfg)  # For debugging without multi-gpu
    
    # Gather all sparse vectors from shards
    ctx_dict = {}
    logger.info("Gathering context sparse vectors from all shards...")
    for i in range(world_size):
        tmp_path_list = glob.glob(os.path.join(cfg.output_dir, f"shard_{i}_*.pkl"))
        tmp_path_list = sorted(tmp_path_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for tmp_path in tmp_path_list:
            with open(tmp_path, 'rb') as f:
                shard_sparse_vectors = pickle.load(f)
                ctx_dict.update(shard_sparse_vectors)
            os.remove(tmp_path)
        logger.info(f"GPU-{i} shards loaded.")
    logger.info(f"Total context vectors: {len(ctx_dict)}")

    # Build Sparse index
    cfg_index = cfg.index
    indexer = SparseIndexer(cfg_index)
    indexer.index_data(ctx_dict, )
    indexer.save(os.path.join(cfg.output_dir, f"{cfg.dataset.name}_sparse"))