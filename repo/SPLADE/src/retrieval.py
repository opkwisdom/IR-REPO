import os
import torch
import hydra
import logging
import json
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from tqdm import tqdm

from model import SpladeLightningModule, Encoder, SpladeEncoder
from indexer import SparseIndexer
from utils import (
    load_queries,
    load_qrels,
    set_seed,
    evaluate_search_results,
    format_results_nested,
    get_best_checkpoint,
    SparseVector
)

# {docid: {token_id: weight, ...}}

logger = logging.getLogger(__file__)

def gen_query_sparse_vectors(
    queries: Dict[str, str],
    encoder: Encoder,
    tokenizer: AutoTokenizer,
    cfg: DictConfig
) -> Tuple[List[str], SparseVector]:
    all_qids = list(queries.keys())
    query_texts = [queries[qid] for qid in all_qids]
    total_len = len(query_texts)
    batch_size = cfg.search.batch_size

    all_doc_ids = []
    all_token_ids = []
    all_scores = []

    iterator = tqdm(range(0, total_len, batch_size), desc="Generating query sparse vectors")
    for start_idx in iterator:
        end_idx = min(total_len, start_idx + batch_size)
        batch_texts = query_texts[start_idx:end_idx]
        batch_ids = all_qids[start_idx:end_idx]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.dataset.max_query_length,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(cfg.device)

        with torch.no_grad():
            sparse_embeddings = encoder(**inputs)
        
        sparse_batch = sparse_embeddings.to_sparse_csr()
        
        crow_indices = sparse_batch.crow_indices().cpu().numpy()
        col_indices = sparse_batch.col_indices().cpu().numpy()
        sparse_values = sparse_batch.values().cpu().numpy()

        cur_batch_size = sparse_embeddings.shape[0]
        for i in range(cur_batch_size):
            start_idx = crow_indices[i]
            row_n_elem = crow_indices[i+1] - crow_indices[i]

            row_qid = batch_ids[i]
            if row_n_elem == 0:
                logger.warning(f"Query {row_qid} doesn't match any other terms.")
                continue
            
            row_token_ids = col_indices[start_idx : start_idx+row_n_elem]
            row_weights = sparse_values[start_idx : start_idx+row_n_elem]
            
            all_doc_ids.append(row_qid)
            all_token_ids.append(row_token_ids.tolist())
            all_scores.append(row_weights.tolist())

    return SparseVector(
        doc_ids=all_doc_ids,
        token_ids=all_token_ids,
        scores=all_scores
    )

@hydra.main(version_base=None, config_path="../conf", config_name="retrieval")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Load queries and qrels
    queries = load_queries(cfg.dataset.queries_path, logger)
    qrels = load_qrels(cfg.dataset.qrels_path, logger)

    # Load Lightning model & tokenizer
    ckpt_path = get_best_checkpoint(cfg.ckpt_dir)
    backbone = SpladeEncoder(cfg.model)
    lightning_module = SpladeLightningModule.load_from_checkpoint(ckpt_path, model=backbone)
    query_encoder = lightning_module.model.query_model.to(cfg.device)
    query_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    query_indices, query_vectors = gen_query_sparse_vectors(queries, query_encoder, tokenizer, cfg)

    # Load Sparse index
    cfg_index = cfg.index
    indexer = SparseIndexer(cfg_index)
    indexer.load(os.path.join(cfg.index_dir, f"{cfg.dataset.name}_sparse"))

    # Search
    top_k = cfg.search.topk
    batch_size = cfg.search.batch_size
    topk_results = indexer.search(query_vectors, query_indices, top_k, batch_size)

    # Evaluate & save results
    eval_results = evaluate_search_results(topk_results, qrels, logger=logger)
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_output_path = os.path.join(cfg.output_dir, "splade_topk_dev.json")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(topk_results, f, indent=4, ensure_ascii=False)

    eval_output_path = os.path.join(cfg.output_dir, "metrics.json")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(format_results_nested(eval_results), f, indent=4)

    logger.info(f"Saved evaluation results to {eval_output_path}")

if __name__ == "__main__":
    main()