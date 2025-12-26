from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer
import logging

from typing import List, Dict
from utils import (
    load_triples,
    load_queries,
    load_qrels,
    load_collection,
    load_dev
)

logger = logging.getLogger(__name__)

class SpladeDataset(Dataset):
    def __init__(self, triple_path: str, queries_path: str, collection: Dict[str, Dict[str, str]]):
        super().__init__()
        self.triples = load_triples(triple_path)
        self.queries = load_queries(queries_path)
        self.collection = collection

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        
        query_text = self.queries[triple.qid]
        pos_doc = self.collection[triple.pos_id]
        neg_doc = self.collection[triple.neg_id]

        return {
            "query_text": query_text,
            "pos_title": pos_doc["title"],
            "pos_contents": pos_doc["contents"],
            "neg_title": neg_doc["title"],
            "neg_contents": neg_doc["contents"],
        }
    
class SpladeDevDataset(Dataset):
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


class SpladeDataModule(pl.LightningDataModule):
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
            self.train_dataset = SpladeDataset(
                triple_path=self.data_cfg.triple_path,
                queries_path=self.data_cfg.queries_path,
                collection=self.collection,
            )
            self.val_dataset = SpladeDevDataset(
                dev_queries_path=self.data_cfg.queries_dev_path,
                dev_qrels_path=self.data_cfg.qrels_dev_path,
                dev_path=self.data_cfg.bm25_dev_path,
                collection=self.collection,
            )
        elif stage == "validate":
            self.val_dataset = SpladeDevDataset(
                dev_queries_path=self.data_cfg.queries_dev_path,
                dev_qrels_path=self.data_cfg.qrels_dev_path,
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
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            collate_fn=self.val_collate_fn,  # Use directly
        )