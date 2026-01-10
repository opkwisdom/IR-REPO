from omegaconf import OmegaConf, DictConfig
from typing import Dict, Tuple, List
import hydra
import logging
import os
import json
from transformers import AutoTokenizer
import numpy as np
import torch
from tqdm import tqdm

from model import DPRLightningModule, Encoder, DPREncoder
from indexer import FaissIndexer
from utils import (
    load_queries,
    load_qrels,
    set_seed,
    evaluate_search_results,
    format_results_nested,
    get_best_checkpoint
)

logger = logging.getLogger(__file__)

def gen_query_vectors(
    queries: Dict[str, str],
    encoder: Encoder,
    tokenizer: AutoTokenizer,
    cfg: DictConfig
) -> Tuple[List[str], np.ndarray]:
    all_qids = list(queries.keys())
    query_texts = [queries[qid] for qid in all_qids]
    total_len = len(query_texts)

    batch_size = cfg.search.batch_size
    query_vectors = []
    query_ids = []

    iterator = tqdm(range(0, total_len, batch_size), desc="Generating query vectors")
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
            embeddings = encoder(**inputs).cpu().numpy()
            query_vectors.append(embeddings)
            query_ids.extend(batch_ids)

    query_vectors = np.vstack(query_vectors)
    return query_ids, query_vectors

@hydra.main(version_base=None, config_path="../conf", config_name="retrieval")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Load queries and qrels
    queries = load_queries(cfg.dataset.queries_path, logger)
    qrels = load_qrels(cfg.dataset.qrels_path, logger)

    # Load Lightning model & tokenizer
    if getattr(cfg, "ckpt_file", None) is None:
        ckpt_path = get_best_checkpoint(cfg.ckpt_dir)
    else:
        ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_file)
    backbone = DPREncoder(cfg.model)
    lightning_module = DPRLightningModule.load_from_checkpoint(ckpt_path, model=backbone)
    query_encoder = lightning_module.model.query_model.to(cfg.device)
    query_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    query_indices, query_vectors = gen_query_vectors(queries, query_encoder, tokenizer, cfg)

    # Load Faiss index
    cfg_index = cfg.index[cfg.index_key]
    indexer = FaissIndexer(cfg_index)
    indexer.load(os.path.join(cfg.index_dir, f"{cfg.dataset.name}_faiss"))

    # Search
    top_k = cfg.search.topk
    batch_size = cfg.search.batch_size
    topk_results = indexer.search(query_vectors, query_indices, top_k, batch_size)

    # Evaluate & save results
    eval_results = evaluate_search_results(topk_results, qrels, logger=logger)
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_output_path = os.path.join(cfg.output_dir, f"{cfg.model.model_type}_dpr_topk_dev.json")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(topk_results, f, indent=4, ensure_ascii=False)

    eval_output_path = os.path.join(cfg.output_dir, f"{cfg.model.model_type}_metrics.json")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(format_results_nested(eval_results), f, indent=4)

    logger.info(f"Saved evaluation results to {eval_output_path}")

if __name__ == "__main__":
    main()