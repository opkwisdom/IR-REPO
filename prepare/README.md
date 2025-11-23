# Preparation of IR Datasets

This directory contains python scripts to download and preprocess various Information Retrieval (IR) datasets used in this project.

The goal is to:
- `download.py`: fetch standard IR benchmarks (e.g. NQ, MS MARCO, TriviaQA, BEIR, etc.)
- `preprocess.py`: normalize them into a common schema for easy consumption during training and evaluation.

## Data schema

All IR datasets in this project (MS MARCO, NQ, TriviaQA, etc.) are normalized into the
following simple schema:

```python
from dataclasses import dataclass

@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: int

@dataclass
class Query:
    id: str
    text: str

@dataclass
class Document:
    id: str
    text: str
    title: str  # may be empty ("") if the source collection has no title
```