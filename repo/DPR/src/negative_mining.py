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
    Query, Qrel, Triple
)

logger = logging.getLogger(__file__)

def mining_negatives(
    cfg: DictConfig,
    searcher: LuceneSearcher,
    queries: Dict[str, str],
    qrels: Dict[str, str],
    logger: logging.Logger = None
) -> List[Triple]:
    triples = []

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
            neg_id = None

            # Sample negative document randomly
            candidate_indices = [
                i for i, hit in enumerate(hits) 
                if hit.docid != pos_id
            ]

            if candidate_indices:
                random_idx = random.choice(candidate_indices)
                neg_id = hits[random_idx].docid
                triples.append(Triple(qid=qid, pos_id=pos_id, neg_id=neg_id))

    logger.info(f"Completed searching all queries, mined {len(triples)} triples.")

    return triples


@hydra.main(version_base=None, config_path="../conf", config_name="negative_mining")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Initialize Lucene Searcher
    searcher = LuceneSearcher(cfg.index.path)
    logger.info(f"Initialized LuceneSearcher with index at {cfg.index.path}")

    # Load datasets
    queries = load_queries(cfg.data.queries_path, logger)
    qrels = load_qrels(cfg.data.qrels_path, logger)

    # Perform search and save results
    search_results = mining_negatives(cfg.search, searcher, queries, qrels, logger)
    output_dir = os.path.join(cfg.search.output_dir, f"{cfg.data.name}.jsonl")
    os.makedirs(cfg.search.output_dir, exist_ok=True)

    with open(output_dir, "w") as f:
        for triple in search_results:
            f.write(json.dumps(asdict(triple)) + "\n")

if __name__ == "__main__":
    main()