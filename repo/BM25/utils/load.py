import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import logging
from tqdm import tqdm

from .structure import Qrel, Query, Document

def load_collection(file_path: str, logger: Optional[logging.Logger] = None) -> List[Document]:
    if logger is not None:
        logger.info(f"Loading collection from {file_path}...")
    
    collection = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            data = json.loads(line)
            doc = Document(**data)
            collection.append(doc)

    if logger is not None:
        logger.info(f"Loaded {len(collection)} documents.")
    return collection

def load_qrels(file_path: str, logger: Optional[logging.Logger] = None) -> List[Qrel]:
    if logger is not None:
        logger.info(f"Loading qrels from {file_path}...")
    
    qrels = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading qrels"):
            data = json.loads(line)
            qrel = Qrel(**data)
            qrels.append(qrel)

    if logger is not None:
        logger.info(f"Loaded {len(qrels)} qrels.")
    return qrels

def load_queries(file_path: str, logger: Optional[logging.Logger] = None) -> List[Query]:
    if logger is not None:
        logger.info(f"Loading queries from {file_path}...")
    
    queries = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading queries"):
            data = json.loads(line)
            query = Query(**data)
            queries.append(query)

    if logger is not None:
        logger.info(f"Loaded {len(queries)} queries.")
    return queries


if __name__ == "__main__":
    doc_path = "/home/ir_repo/work/hdd/data/preprocessed/msmarco_collection/collection.jsonl"
    query_path = "/home/ir_repo/work/hdd/data/preprocessed/datasets/msmarco/dev_queries.jsonl"
    qrel_path = "/home/ir_repo/work/hdd/data/preprocessed/datasets/msmarco/dev_qrels.jsonl"
    
    # docs = load_collection(doc_path)
    queries = load_queries(query_path)
    qrels = load_qrels(qrel_path)
    for q in queries[:5]:
        print(q)