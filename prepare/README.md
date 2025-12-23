# 정보 검색(IR) 데이터셋 준비

이 디렉토리에는 본 프로젝트에서 사용되는 다양한 정보 검색(IR) 데이터셋(예: NQ, MS MARCO, TriviaQA, BEIR 등)을 다운로드하고 전처리하기 위한 파이썬 스크립트들이 포함되어 있습니다.

주요 목표는 다음과 같습니다:
- `download.py`: 표준 IR 벤치마크 데이터셋 다운로드 (e.g. NQ, MS MARCO, TriviaQA, BEIR, etc.)
- `preprocess.py`: 훈련 및 평가 시 쉽게 사용할 수 있도록 해당 데이터셋들을 공통 스키마로 규격화

## Data schema

본 프로젝트의 모든 IR 데이터셋(MS MARCO, NQ, TriviaQA 등)은 다음의 간단한 스키마로 정규화됩니다:

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
    contents: str
    title: str  # may be empty ("") if the source collection has no title
```