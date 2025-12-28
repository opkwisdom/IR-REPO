import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm
from collections import defaultdict
import glob
import re

from .structure import Qrel, Query, Document, Triple, TripleCandidates

### Loading collection, queries, qrels
def load_collection(file_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Dict[str, str]]:
    if logger is not None:
        logger.info(f"Loading collection from {file_path}...")
    
    collection = {}
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            data = json.loads(line)
            collection[data['id']] = {
                "title": data["title"],
                "contents": data["contents"]
            }

    if logger is not None:
        logger.info(f"Loaded {len(collection)} documents.")
    return collection

def load_qrels(file_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, List[str]]:
    if logger is not None:
        logger.info(f"Loading qrels from {file_path}...")
    
    qrels = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading qrels"):
            data = json.loads(line)
            if data['relevance'] > 0:
                qrels[data['query_id']].append(data['doc_id'])

    if logger is not None:
        logger.info(f"Loaded {len(qrels)} qrels.")
    return qrels

def load_queries(file_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    if logger is not None:
        logger.info(f"Loading queries from {file_path}...")
    
    queries = {}
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading queries"):
            data = json.loads(line)
            queries[data['id']] = data['text']

    if logger is not None:
        logger.info(f"Loaded {len(queries)} queries.")
    return queries

### Loading triples for training
def load_triples(file_path: str, logger: Optional[logging.Logger] = None) -> List[Triple]:
    if logger is not None:
        logger.info(f"Loading triples from {file_path}...")
    
    triples = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading triples"):
            data = json.loads(line)
            triple = Triple(**data)
            triples.append(triple)

    if logger is not None:
        logger.info(f"Loaded {len(triples)} triples.")
    return triples


### Loading dev set, for evaluation
# This dataset would be naive dataset, containing queries and Top-1000 bm25 retrieved docs
def load_dev(file_path: str, logger: Optional[logging.Logger] = None) -> List[Dict[str, List[str]]]:
    if logger is not None:
        logger.info(f"Loading dev set from {file_path}...")
    
    with open(file_path, 'r') as f:
        dev_data = json.load(f)
    # Format compatibility
    dev_data = [
        {k: v} for k, v in dev_data.items()
    ]

    if logger is not None:
        logger.info(f"Loaded {len(dev_data)} dev entries.")
    return dev_data

def load_triple_candidates(file_path: str, logger: Optional[logging.Logger] = None) -> List[TripleCandidates]:
    if logger is not None:
        logger.info(f"Loading triple candidates from {file_path}...")

    triple_candidates = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading triple candidates"):
            data = json.loads(line)
            triple = TripleCandidates(**data)
            triple_candidates.append(triple)

    if logger is not None:
        logger.info(f"Loaded {len(triple_candidates)} triple candidates.")
    return triple_candidates

def get_best_checkpoint(checkpoint_dir: str, metric: str = "val_mrr_10") -> str:
    best_score = float("-inf")
    best_checkpoint = None
    
    checkpoint_paths = glob.glob(f"{checkpoint_dir}/*.ckpt")
    pattern = re.compile(rf".*{re.escape(metric)}=([0-9]+(?:\.[0-9]+)?)")
    
    for path in checkpoint_paths:
        match = pattern.search(path)
        if match:
            score = float(match.group(1))
            if score > best_score:
                best_score = score
                best_checkpoint = path
    return best_checkpoint