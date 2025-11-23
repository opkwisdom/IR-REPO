from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Triple:
    qid: str
    pos_id: str
    neg_ids: List[str] = field(default_factory=list)

