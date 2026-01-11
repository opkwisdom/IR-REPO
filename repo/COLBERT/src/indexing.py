from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Tuple, List, Union
import hydra
import logging
import os
from transformers import AutoTokenizer
import torch
import numpy as np
import pickle
import torch.multiprocessing as mp
from tqdm import tqdm

from model import ColBERTLightningModule, Encoder, ColBERTEncoder
from indexer import ColBERTIndexer, IndexFileManager
from utils import (
    load_collection,
    set_seed,
    setup, cleanup,
    get_best_checkpoint
)

logger = logging.getLogger(__file__)
CHUNK_N_EMBS_THRESHOLD = 100_000

### ================ Reference ================ ###
def doc_tokenize(tokenizer: AutoTokenizer, doc_texts: List[str], d_marker_id: int, max_passage_length: int):
    """
    Same as `doc_tokenize` method in datamodule
    """
    d_ids = tokenizer(
        doc_texts,
        padding=False,
        truncation=True,
        max_length=max_passage_length - 3,
        add_special_tokens=False,
        return_token_type_ids=False,
    )["input_ids"]
    
    # Refer ColBERT github implementation
    prefix, postfix = [tokenizer.cls_token_id, d_marker_id], [tokenizer.sep_token_id]
    padded_d_ids = []
    attention_mask = []
    for d in d_ids:
        base_ids = prefix + d + postfix
        pad_len = max(0, max_passage_length - len(base_ids))
        full_ids = base_ids + [tokenizer.pad_token_id] * pad_len
        mask = [1] * len(base_ids) + [0] * pad_len
        padded_d_ids.append(full_ids)
        attention_mask.append(mask)
    input_ids = torch.tensor(padded_d_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
### ================ Reference ================ ###

def _get_file_output_paths(cfg: DictConfig, rank: int, file_postfix: int) -> Tuple[str, str]:
    ids_path = os.path.join(cfg.output_dir, f"shard_{rank}_{file_postfix}.pkl")
    emb_path = os.path.join(cfg.output_dir, f"shard_{rank}_{file_postfix}.npy")
    return ids_path, emb_path


def gen_ctx_vectors(
    rank: int,
    shard_collection: Dict[str, Dict[str, str]],
    encoder: Encoder,
    tokenizer: AutoTokenizer,
    cfg: DictConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate ctx vectors, each GPU process, then save chunks
    """
    all_pids = list(shard_collection.keys())
    all_texts = [shard_collection[pid] for pid in all_pids]
    total_len = len(all_texts)
    logger.info(f"Generating context vectors for {total_len} passages...")
    
    batch_size = cfg.search.batch_size
    ctx_vectors = []
    ctx_ids = []
    ctx_doclens = []

    file_postfix = 0
    cur_n_embs = 0
    iterator = tqdm(range(0, total_len, batch_size), desc=f"GPU {rank}", position=rank)
    d_marker_id = tokenizer.convert_tokens_to_ids("[D]")
    for start_idx in iterator:
        end_idx = min(total_len, start_idx + batch_size)
        batch_ids = all_pids[start_idx:end_idx]
        batch_data = all_texts[start_idx:end_idx]
        batch_psgs = [
            d["title"] + " " + d["contents"]
            for d in batch_data
        ]
        batch_inputs = doc_tokenize(tokenizer, batch_psgs, d_marker_id, cfg.dataset.max_passage_length)
        batch_inputs = {k: v.to(encoder.model.device) for k, v in batch_inputs.items()}
        
        with torch.no_grad():
            ctx_embs = encoder(**batch_inputs)    # (B, L_p, D)
            mask = batch_inputs["attention_mask"].bool()
            valid_embs = ctx_embs[mask].cpu().numpy()     # (valid, D)
        
        # Compute doclens to ensure one-to-many mapping
        batch_doclens = torch.sum(mask, dim=1).tolist()
        
        ctx_vectors.append(valid_embs)
        ctx_ids.extend(batch_ids)
        ctx_doclens.extend(batch_doclens)
        cur_n_embs += valid_embs.shape[0]
        
        # Save current chunk
        if cur_n_embs >= CHUNK_N_EMBS_THRESHOLD:
            logger.info(f"Save Chunk {rank}-{file_postfix}")
            ids_path, emb_path = _get_file_output_paths(cfg, rank, file_postfix)
            # Save ids & doclens
            pkl_data = {"pids": ctx_ids, "doclens": ctx_doclens}
            with open(ids_path, 'wb') as f:
                pickle.dump(pkl_data, f)
            # Save embs
            ctx_vectors = np.vstack(ctx_vectors)    # (N, L_p, D)
            np.save(emb_path, ctx_vectors)
            
            cur_n_embs = 0
            file_postfix += 1
            ctx_vectors = []
            ctx_ids = []
            ctx_doclens = []
    
    # Save last chunk
    if ctx_ids:
        logger.info(f"Save Chunk {rank}-{file_postfix}")
        ids_path, emb_path = _get_file_output_paths(cfg, rank, file_postfix)
        # Save ids & doclens
        pkl_data = {"pids": ctx_ids, "doclens": ctx_doclens}
        with open(ids_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        # Save embs
        ctx_vectors = np.vstack(ctx_vectors)    # (N, L_p, D)
        np.save(emb_path, ctx_vectors)
    logger.info("Finish generate ctx vectors.")


def multi_gpu_worker(rank: int, world_size: int, collection_shards: List[Dict[str, Dict[str, str]]], cfg: DictConfig):
    torch.cuda.set_device(rank)
    setup(rank, world_size)
    
    # Load Lightning model & tokenizer
    try:
        ckpt_path = get_best_checkpoint(cfg.ckpt_dir)
        
        # Prepare the interfaces
        backbone_model = ColBERTEncoder(cfg.model)
        backbone_tok = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
        special_tokens = ["[Q]", "[D]"]
        backbone_tok.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        backbone_model.resize_token_embeddings(len(backbone_tok))
        
        lightning_module = ColBERTLightningModule.load_from_checkpoint(ckpt_path, model=backbone_model, tokenizer=backbone_tok)
        context_encoder = lightning_module.model.context_model.to(rank)
        context_encoder.eval()
        tokenizer = lightning_module.tokenizer
        logger.info(f"Vocab size: {len(tokenizer)}")
        gen_ctx_vectors(rank, collection_shards[rank], context_encoder, tokenizer, cfg)
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
    
    # Multi-GPU indexing
    os.makedirs(cfg.output_dir, exist_ok=True)
    mp.spawn(multi_gpu_worker, args=(world_size, shards, cfg), nprocs=world_size, join=True)
    # multi_gpu_worker(0, world_size, shards, cfg)  # For debugging without multi-gpu
    
    # Instantiate Index File Manager & ColBERTIndexer
    file_manager = IndexFileManager(cfg.output_dir)
    cfg_index = cfg.index[cfg.index_key]
    cfg_index.output_dir = cfg.output_dir
    indexer = ColBERTIndexer(cfg_index)
    
    # Build FAISS index
    logger.info("Stage 1: Centroid selection by Kmeans")
    codec_dir = f"{cfg.output_dir}/codec"
    sampled_vectors, num_partitions = file_manager.sample_vectors(cfg_index.k, cfg.seed)
    indexer.train(sampled_vectors, num_partitions, codec_dir)
    
    logger.info("Stage 2: Passage Encoding")
    iterator = file_manager.stream_batches(cfg_index.stream_bsize)
    indexer.index_data(iterator)
    logger.info("Indexing completed")
    
    file_manager.finalize()
    
if __name__ == "__main__":
    main()