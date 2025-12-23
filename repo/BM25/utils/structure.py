from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Base data structures
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
    contents: str
    title: str  # may be empty ("") if the source collection has no title