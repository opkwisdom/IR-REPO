from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Tuple, List
import hydra
import logging
import os
from transformers import AutoTokenizer
import torch
import numpy as np
import json
import pickle
from tqdm import tqdm

from model import ColBERTLightningModule, Encoder, ColBERTEncoder
from indexer import ColBERTIndexer
from utils import (
    load_collection,
    load_queries,
    load_qrels,
    set_seed,
    get_best_checkpoint,
    evaluate_search_results,
    format_results_nested
)

logger = logging.getLogger(__file__)

### ================ Reference ================ ###
def query_tokenize(tokenizer: AutoTokenizer, query_texts: List[str], q_marker_id: int, max_query_length: int):
    """
    Tokenize query texts for ColBERT.
    All query inputs are used. (Not padded, just masked, query augumentation)
    """
    q_ids = tokenizer(
        query_texts,
        padding=False,
        truncation=True,
        max_length=max_query_length - 3,
        add_special_tokens=False,
        return_token_type_ids=False
    )["input_ids"]
    
    # Refer ColBERT github implementation
    prefix, postfix = [tokenizer.cls_token_id, q_marker_id], [tokenizer.sep_token_id]
    padded_q_ids = []
    for q in q_ids:
        base_ids = prefix + q + postfix
        pad_len = max(0, max_query_length - len(base_ids))
        full_ids = base_ids + [tokenizer.mask_token_id] * pad_len
        padded_q_ids.append(full_ids)
    input_ids = torch.tensor(padded_q_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
### ================ Reference ================ ###

def gen_query_vectors(
    queries: Dict[str, str],
    encoder: Encoder,
    tokenizer: AutoTokenizer,
    cfg: DictConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate query vectors
    """
    all_qids = list(queries.keys())
    query_texts = [queries[qid] for qid in all_qids]
    total_len = len(query_texts)

    batch_size = cfg.search.batch_size
    query_vectors = []
    query_ids = []

    iterator = tqdm(range(0, total_len, batch_size), desc="Generating query vectors")
    q_marker_id = tokenizer.convert_tokens_to_ids("[Q]")
    for start_idx in iterator:
        end_idx = min(total_len, start_idx + batch_size)
        batch_ids = all_qids[start_idx:end_idx]
        batch_data = query_texts[start_idx:end_idx]
        
        batch_inputs = query_tokenize(tokenizer, batch_data, q_marker_id, cfg.dataset.max_query_length)
        batch_inputs = {k: v.to(encoder.model.device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            embeddings = encoder(**batch_inputs).cpu().numpy()  # (B, L_q, D)
            query_vectors.append(embeddings)
            query_ids.extend(batch_ids)

    query_vectors = np.vstack(query_vectors)    # (N, L_q, D)
    return query_vectors, query_ids
    
@hydra.main(version_base=None, config_path="../conf", config_name="retrieval")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Load queries and qrels
    queries = load_queries(cfg.dataset.queries_path, logger)
    qrels = load_qrels(cfg.dataset.qrels_path, logger)

    # Load Lightning model & tokenizer
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
    query_encoder = lightning_module.model.query_model.to(cfg.device)
    query_encoder.eval()
    tokenizer = lightning_module.tokenizer
    logger.info(f"Vocab size: {len(tokenizer)}")
    
    query_vectors, query_indices = gen_query_vectors(queries, query_encoder, tokenizer, cfg)
    
    # Load ColBERT index
    cfg_index = cfg.index[cfg.index_key]
    codec_dir = f"{cfg.output_dir}/codec"
    indexer = ColBERTIndexer(cfg_index)
    indexer.load_codec(codec_dir)
    
    # Search
    top_k = cfg.search.topk
    batch_size = cfg.search.batch_size
    topk_results = indexer.search(query_vectors, query_indices, top_k, batch_size)
    
    # Evaluate & save results
    eval_results = evaluate_search_results(topk_results, qrels, logger=logger)
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_output_path = os.path.join(cfg.output_dir, f"{cfg.model.model_type}_colbert_v1_topk_dev.json")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(topk_results, f, indent=4, ensure_ascii=False)

    eval_output_path = os.path.join(cfg.output_dir, f"{cfg.model.model_type}_metrics.json")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(format_results_nested(eval_results), f, indent=4)

    logger.info(f"Saved evaluation results to {eval_output_path}")

if __name__ == "__main__":
    main()