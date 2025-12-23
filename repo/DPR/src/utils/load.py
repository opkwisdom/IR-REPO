import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm

from .structure import Qrel, Query, Document, Triple

### Loading collection, queries, qrels
def load_collection(file_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    if logger is not None:
        logger.info(f"Loading collection from {file_path}...")
    
    collection = {}
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            data = json.loads(line)
            contents = f"Title: {data['title']}\n\n{data['contents']}" if data['title'] else data['contents']
            collection[data['id']] = contents

    if logger is not None:
        logger.info(f"Loaded {len(collection)} documents.")
    return collection

def load_qrels(file_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    if logger is not None:
        logger.info(f"Loading qrels from {file_path}...")
    
    qrels = {}
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading qrels"):
            data = json.loads(line)
            if data['relevance'] > 0:
                qrels[data['query_id']] = data['doc_id']

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