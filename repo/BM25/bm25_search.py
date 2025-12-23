from omegaconf import DictConfig, OmegaConf
from pyserini.search.lucene import LuceneSearcher
import logging
import hydra
import json
import os
from typing import List, Dict
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig

from utils import (
    evaluate_search_results,
    format_results_nested,
    load_queries,
    load_qrels,
    Query
)

logger = logging.getLogger(__file__)

def search_documents(
    cfg: DictConfig,
    searcher: LuceneSearcher,
    queries: List[Query],
    logger: logging.Logger = None
) -> Dict[str, List[str]]:
    results = {}

    topk = cfg.topk
    batch_size = cfg.batch_size
    num_queries = len(queries)
    logger.info(f"Searching {num_queries} queries with batch size {batch_size} and topk {topk}")

    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Searching queries"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_queries = queries[start_idx:end_idx]
        query_texts = [q.text for q in batch_queries]
        qids = [q.id for q in batch_queries]

        # Perform batch search
        batch_search_res = searcher.batch_search(query_texts, qids, k=topk) # (B, topk)
        for qid, hits in batch_search_res.items():
            results[qid] = []
            for hit in hits:
                results[qid].append(hit.docid)
    logger.info("Completed searching all queries.")

    return results


@hydra.main(version_base=None, config_path="conf", config_name="bm25")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Initialize Lucene Searcher
    searcher = LuceneSearcher(cfg.index.path)
    logger.info(f"Initialized LuceneSearcher with index at {cfg.index.path}")

    # Load datasets
    queries = load_queries(cfg.data.queries_path, logger)
    qrels = load_qrels(cfg.data.qrels_path, logger)

    # Perform search and save results
    search_results = search_documents(cfg.search, searcher, queries, logger)

    # Save results to output file
    eval_results = evaluate_search_results(search_results, qrels, logger=logger)
    output_dir = HydraConfig.get().runtime.output_dir
    eval_output_path = os.path.join(output_dir, "metrics.json")
    
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        json.dump(format_results_nested(eval_results), f, indent=4)

    logger.info(f"Saved evaluation results to {eval_output_path}")

if __name__ == "__main__":
    main()