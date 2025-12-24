from typing import List, Dict
import logging
import math
from collections import defaultdict

def evaluate_search_results(
    search_results: Dict[str, List[str]],
    qrels: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10, 20, 50, 100, 1000],
    logger: logging.Logger = None
) -> Dict[str, float]:
    """
    Evaluate search results using simple metrics
    - Recall@k
    - nDCG@k
    - MRR@k
    """
    metrics = {}
    for k in k_values:
        metrics[f'Recall@{k}'] = []
        metrics[f'nDCG@{k}'] = []
        metrics[f'MRR@{k}'] = []

    for qid, doc_ids in search_results.items():
        if qid not in qrels:
            continue
        
        relevant_docs = qrels[qid]

        for k in k_values:
            top_k_hits = doc_ids[:k]

            # Calculate Recall@k
            num_rel_retrieved = sum([1 for doc_id in top_k_hits if doc_id in relevant_docs])
            recall = num_rel_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
            metrics[f'Recall@{k}'].append(recall)

            # Calculate nDCG@k
            dcg = 0.0
            idcg = 0.0

            # DCG calculation
            for i, doc_id in enumerate(top_k_hits):
                if doc_id in relevant_docs:
                    dcg += 1.0 / (math.log2(i + 2))
            
            # IDCG calculation
            num_rel = len(relevant_docs)
            for i in range(min(num_rel, k)):
                idcg += 1.0 / (math.log2(i + 2))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'nDCG@{k}'].append(ndcg)

            # Calculate MRR@k
            mrr = 0.0
            for i, doc_id in enumerate(top_k_hits):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (i + 1)
                    break   # First relevant document found
            metrics[f'MRR@{k}'].append(mrr)
    
    # Aggregate metrics
    final_metrics = {k: (sum(v) / len(v) if len(v) > 0 else 0.0) for k, v in metrics.items()}
    if logger:
        logger.info("Evaluation Results:")
        for metric, value in final_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    return final_metrics

def format_results_nested(results: Dict[str, float]) -> Dict:
    nested = {}
    for key, val in results.items():
        metric, k = key.split('@')
        if metric not in nested:
            nested[metric] = {}
        nested[metric][k] = round(val, 4) # 소수점 정리
    return nested