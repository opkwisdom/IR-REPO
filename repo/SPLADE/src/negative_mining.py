from omegaconf import DictConfig, OmegaConf
from pyserini.search.lucene import LuceneSearcher
import logging
import hydra
import json
import os
import random
from dataclasses import asdict
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict

from utils import (
    load_queries,
    load_qrels,
    set_seed,
    Query, Qrel, TripleCandidates
)

logger = logging.getLogger(__file__)

def mining_negatives(
    cfg: DictConfig,
    searcher: LuceneSearcher,
    queries: Dict[str, str],
    qrels: Dict[str, List[str]],
    logger: logging.Logger = None
) -> List[TripleCandidates]:
    triple_candidates = []

    topk = cfg.topk
    batch_size = cfg.batch_size

    qid_list = list(queries.keys())
    num_queries = len(qid_list)
    
    if logger:
        logger.info(f"Searching {num_queries} queries with batch size {batch_size} and topk {topk}")

    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Searching queries"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_qids = qid_list[start_idx:end_idx]
        query_texts = [queries[qid] for qid in batch_qids]

        # Perform batch search
        batch_search_res = searcher.batch_search(query_texts, batch_qids, k=topk) # (B, topk)
        for qid, hits in batch_search_res.items():
            pos_id = qrels[qid]  # Take the first relevant doc as positive
            neg_ids = [
                hit.docid for hit in hits if hit.docid not in pos_id
            ]
            triple_candidates.append(
                TripleCandidates(qid=qid, pos_id=pos_id[0], neg_ids=neg_ids)
            )

    logger.info(f"Completed searching all queries, mined {len(triple_candidates)} triples.")

    return triple_candidates


@hydra.main(version_base=None, config_path="../conf", config_name="negative_mining")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Initialize Lucene Searcher
    cfg_indexer = cfg.index[cfg.index_key]
    searcher = LuceneSearcher(cfg_indexer.path)
    logger.info(f"Initialized LuceneSearcher with index at {cfg_indexer.path}")

    if cfg.dataset.name == "msmarco":
        k1 = 0.82
        b = 0.68
        searcher.set_bm25(k1=k1, b=b)
        logger.info(f"Set BM25 parameters: k1={k1}, b={b}")
    
    # Load datasets
    queries = load_queries(cfg.dataset.queries_path, logger)
    qrels = load_qrels(cfg.dataset.qrels_path, logger)

    # Perform search and save results
    search_results = mining_negatives(cfg.search, searcher, queries, qrels, logger)
    output_dir = os.path.join(cfg.output_dir, f"{cfg.dataset.name}.jsonl")
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(output_dir, "w") as f:
        for triple in search_results:
            f.write(json.dumps(asdict(triple)) + "\n")

if __name__ == "__main__":
    main()