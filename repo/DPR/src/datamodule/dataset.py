from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer

from typing import List
from utils import (
    load_triples,
    load_queries,
    load_collection,
)

class DPRDataset(Dataset):
    def __init__(self, triple_path: str, queries_path: str, collection_path: str):
        super().__init__()
        self.triples = load_triples(triple_path)
        self.queries = load_queries(queries_path)
        self.collection = load_collection(collection_path)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        
        query_text = self.queries[triple.qid]
        pos_doc_text = self.collection[triple.pos_id]
        neg_doc_text = self.collection[triple.neg_id]

        return {
            "query_text": query_text,
            "pos_doc_text": pos_doc_text,
            "neg_doc_text": neg_doc_text,
        }


class DPRDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer: AutoTokenizer):
        super().__init__()
        self.data_cfg = cfg.dataset
        self.train_cfg = cfg.train
        self.seed = cfg.seed
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = DPRDataset(
                triple_path=self.data_cfg.triple_path,
                queries_path=self.data_cfg.queries_path,
                collection_path=self.data_cfg.collection_path,
            )
            train_size = int((1 - self.data_cfg.test_size) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

    def collate_fn(self, batch):
        query_texts = [item["query_text"] for item in batch]
        pos_doc_texts = [item["pos_doc_text"] for item in batch]
        neg_doc_texts = [item["neg_doc_text"] for item in batch]

        q_inputs = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            max_length=self.data_cfg.max_query_length,
            return_tensors="pt",
            return_token_type_ids=False
        )
        p_inputs = self.tokenizer(
            pos_doc_texts + neg_doc_texts,
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

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg.per_device_batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_cfg.per_device_batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


### Test code
if __name__ == "__main__":
    triple_path = "/home/ir_repo/work/repo/DPR/src/hard_negatives/msmarco.jsonl"
    queries_path = "/home/ir_repo/work/hdd/data/preprocessed/datasets/msmarco/train_queries.jsonl"
    collection_path = "/home/ir_repo/work/hdd/data/preprocessed/msmarco_collection/collection.jsonl"
    dataset = DPRDataset(
        triple_path=triple_path,
        queries_path=queries_path,
        collection_path=collection_path,
    )

    print(len(dataset))
    print(dataset[0])