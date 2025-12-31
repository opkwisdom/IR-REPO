from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer
from typing import Dict
import logging
import random
import sys
import csv

from typing import List
from utils import (
    load_triple_candidates,
    load_queries,
    load_qrels,
    load_dev,
    load_collection,
)

logger = logging.getLogger(__name__)

class DPRDataset(Dataset):
    def __init__(self, triple_path: str, queries_path: str, collection: Dict[str, Dict[str, str]], n_negative: int = 50):
        super().__init__()
        self.triple_candidates = load_triple_candidates(triple_path)
        self.queries = load_queries(queries_path)
        self.collection = collection
        self.n_negative = n_negative

    def __len__(self):
        return len(self.triple_candidates)

    def __getitem__(self, idx):
        triple = self.triple_candidates[idx]

        query_text = self.queries[triple.qid]
        pos_doc = self.collection[triple.pos_id]

        topk_neg_ids = triple.neg_ids[:self.n_negative]
        neg_id = random.choice(topk_neg_ids)
        neg_doc = self.collection[neg_id]

        return {
            "query_text": query_text,
            "pos_title": pos_doc["title"],
            "pos_contents": pos_doc["contents"],
            "neg_title": neg_doc["title"],
            "neg_contents": neg_doc["contents"],
        }
    

class DPRMSMarcoDataset(IterableDataset):
    def __init__(self, triple_path: str, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.triple_path = triple_path
        self.rank = rank
        self.world_size = world_size

        csv.field_size_limit(sys.maxsize)

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            num_workers_per_gpu = 1
            worker_id = 0
        else:
            num_workers_per_gpu = worker_info.num_workers
            worker_id = worker_info.id
        
        global_worker_id = self.rank * num_workers_per_gpu + worker_id
        total_workers = self.world_size * num_workers_per_gpu

        with open(self.triple_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # Read only lines assigned to this worker
                if i % total_workers != global_worker_id:
                    continue
                query_text, pos_text, neg_text = line.strip().split('\t')
                yield {
                    "query_text": query_text,
                    "pos_title": "",
                    "pos_contents": pos_text,
                    "neg_title": "",
                    "neg_contents": neg_text,
                }


class DPRDevDataset(Dataset):
    def __init__(self, dev_queries_path: str, dev_qrels_path: str, dev_path: str, collection: Dict[str, Dict[str, str]]):
        super().__init__()
        self.dev_queries = load_queries(dev_queries_path)
        self.dev_qrels = load_qrels(dev_qrels_path)
        self.dev_data = load_dev(dev_path)
        self.collection = collection
    
    def get_topk_dev(self):
        return self.dev_data

    def get_qrels(self):
        return self.dev_qrels

    def __len__(self):
        return len(self.dev_data)
    
    def __getitem__(self, idx):
        entry = self.dev_data[idx]
        query_id, passage_ids = list(entry.items())[0]
        return {
            "query_id": query_id,
            "passage_ids": passage_ids,
            "query_text": self.dev_queries[query_id],
            "passage_titles": [self.collection[idx]["title"] for idx in passage_ids],
            "passage_contents": [self.collection[idx]["contents"] for idx in passage_ids],
        }


class DPRDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.data_cfg = cfg.dataset
        self.train_cfg = cfg.train
        self.seed = cfg.seed
        self.tokenizer = tokenizer

        self.collection = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.collection is None:
            logger.info("Loading collection...")
            self.collection = load_collection(self.data_cfg.collection_path)
            logger.info("Collection loaded.")

        if stage == "fit" or stage is None:
            if self.data_cfg.use_msmarco:
                if not self.data_cfg.triple_path.endswith(".tsv"):
                    raise ValueError("For MSMarco dataset, triple_path should be a .tsv file.")
                self.train_dataset = DPRMSMarcoDataset(
                    triple_path=self.data_cfg.triple_path,
                    rank=self.trainer.global_rank,
                    world_size=self.trainer.world_size
                )
                logger.info("Using MSMarco Iterable Dataset for training.")
            else:
                if not self.data_cfg.triple_path.endswith(".jsonl"):
                    raise ValueError("For standard DPR dataset, triple_path should be a .jsonl file.")
                self.train_dataset = DPRDataset(
                    triple_path=self.data_cfg.triple_path,
                    queries_path=self.data_cfg.queries_path,
                    collection=self.collection,
                    n_negative=self.train_cfg.n_negative,
                )
                logger.info("Using standard Map-style Dataset for training.")

            self.val_dataset = DPRDevDataset(
                dev_queries_path=self.data_cfg.dev_queries_path,
                dev_qrels_path=self.data_cfg.dev_qrels_path,
                dev_path=self.data_cfg.bm25_dev_path,
                collection=self.collection,
            )

    def train_collate_fn(self, batch):
        query_texts = [item["query_text"] for item in batch]

        # Bert-style input: [CLS] title [SEP] contents [SEP]
        pos_titles = [item["pos_title"] for item in batch]
        pos_contents = [item["pos_contents"] for item in batch]
        neg_titles = [item["neg_title"] for item in batch]
        neg_contents = [item["neg_contents"] for item in batch]

        q_inputs = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=self.data_cfg.max_query_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        p_inputs = self.tokenizer(
            text=pos_titles + neg_titles,
            text_pair=pos_contents + neg_contents,
            padding=True,
            truncation=True,
            max_length=self.data_cfg.max_passage_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        return {
            "queries": q_inputs,    # input_ids, attention_mask
            "passages": p_inputs,
        }

    def val_collate_fn(self, batch):
        batch = batch[0]    # unpack
        
        q_inputs = self.tokenizer(
            [batch["query_text"]],
            padding=True,
            truncation=True,
            max_length=self.data_cfg.max_query_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        p_inputs = self.tokenizer(
            text=batch["passage_titles"],
            text_pair=batch["passage_contents"],
            padding=True,
            truncation=True,
            max_length=self.data_cfg.max_passage_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        return {
            "query_id": batch["query_id"],
            "passage_ids": batch["passage_ids"],
            "queries": q_inputs,    # input_ids, attention_mask
            "passages": p_inputs,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg.per_device_batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            collate_fn=self.train_collate_fn,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            collate_fn=self.val_collate_fn,
            drop_last=False,
        )