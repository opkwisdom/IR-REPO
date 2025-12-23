from omegaconf import DictConfig, OmegaConf
from pyserini.search.lucene import LuceneSearcher
import logging
import hydra
import json
import os
import random
from dataclasses import asdict
from typing import List
from tqdm import tqdm
from collections import defaultdict

from utils import (
    load_queries,
    load_qrels,
    set_seed,
    Query, Qrel, Triple
)
from model import DPREncoder

logger = logging.getLogger(__file__)

@hydra.main(version_base=None, config_path="../conf", config_name="dpr_train")
def main(cfg: DictConfig):
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Initialize searcher
    searcher = LuceneSearcher(cfg.search.index_path)

    # Load queries and qrels
    queries = load_queries(cfg.data.queries_path, logger)


if __name__ == "__main__":
    main()