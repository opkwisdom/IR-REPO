import json
import os
from tqdm import tqdm
import glob
import sys
from dataclasses import dataclass, asdict
from collections import defaultdict

BASE_INPUT_DIR = "/home/ir_repo/work/hdd/data/raw"
BASE_OUTPUT_DIR = "/home/ir_repo/work/hdd/data/preprocessed"

### =================================== DATA CLASSES ========================================= ###
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
    title: str
### ========================================================================================== ###

### ==================================== PREPROCESS ========================================== ###
def preprocess_collections() -> None:
    """Preprocess document collections."""
    print("🛠️ Preprocessing document collections...")
    collection_list = ["msmarco_collection.tsv", "psgs_w100.tsv"]
    output_dir = os.path.join(BASE_OUTPUT_DIR, "collection")
    os.makedirs(output_dir, exist_ok=True)
    
    for collection_file in collection_list:
        collection_input_path = os.path.join(BASE_INPUT_DIR, "collection", collection_file)
        collection_output_path = os.path.join(output_dir, collection_file.replace(".tsv", ".jsonl"))

        with open(collection_input_path, 'r', encoding='utf-8') as f_in, \
             open(collection_output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc=f"Processing {collection_file} collection"):
                parts = line.rstrip('\n').split('\t')
                
                # Handle different formats
                if len(parts) == 2:
                    doc_id, text = parts
                    title = ""
                elif len(parts) == 3:
                    doc_id, text, title = parts
                else:
                    raise ValueError(f"Unexpected number of columns in line: {line}")
                document = Document(id=doc_id, contents=text, title=title)
                f_out.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    
    print("Document collections preprocessing completed.")


def preprocess_msmarco() -> None:
    """Preprocess MSMARCO datasets."""
    print("🛠️ Preprocessing MSMARCO datasets...")
    split_list = ["train", "dev"]
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, "datasets", "msmarco"), exist_ok=True)
    
    for _split in split_list:
        # Preprocess queries
        query_input_path = os.path.join(BASE_INPUT_DIR, "datasets", "msmarco", f"queries.{_split}.tsv")
        query_output_path = os.path.join(BASE_OUTPUT_DIR, "datasets", "msmarco", f"{_split}_queries.jsonl")
        with open(query_input_path, 'r', encoding='utf-8') as f_in, \
            open(query_output_path, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, desc=f"Processing MSMARCO {_split} queries"):
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) != 2:
                        raise ValueError(f"Unexpected number of columns in line: {line}")
                    query_id, question = parts
                    query = Query(id=query_id, text=question)
                    f_out.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")
        
        # Preprocess qrels
        qrel_input_path = os.path.join(BASE_INPUT_DIR, "datasets", "msmarco", f"{_split}_qrels.tsv")
        qrel_output_path = os.path.join(BASE_OUTPUT_DIR, "datasets", "msmarco", f"{_split}_qrels.jsonl")
        with open(qrel_input_path, 'r', encoding='utf-8') as f_in, \
            open(qrel_output_path, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, desc=f"Processing MSMARCO {_split} qrels"):
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        raise ValueError(f"Unexpected number of columns in line: {line}")
                    query_id, _, doc_id, relevance = parts
                    relevance = int(relevance)
                    if relevance < 1:
                        continue    # Skip non-relevant documents
                    qrel = Qrel(query_id=query_id, doc_id=doc_id, relevance=relevance)
                    f_out.write(json.dumps(asdict(qrel), ensure_ascii=False) + "\n")

    print("MSMARCO preprocessing completed.")


def preprocess_ir_datasets() -> None:
    """Preprocess IR Datasets."""
    print("🛠️ Preprocessing IR Datasets...")
    # dataset_list = ["nq", "triviaqa", "msmarco"]
    dataset_list = ["msmarco"]
    split_list = ["train", "dev"]
    
    for dataset in dataset_list:
        print(f"Preprocessing {dataset}...")
        data_dir = os.path.join(BASE_INPUT_DIR, "datasets", dataset)
        output_dir = os.path.join(BASE_OUTPUT_DIR, "datasets", dataset)
        os.makedirs(output_dir, exist_ok=True)

        for _split in split_list:
            # Preprocess queries
            query_input_path = os.path.join(data_dir, f"{_split}_queries.jsonl")
            query_output_path = os.path.join(output_dir, f"{_split}_queries.jsonl")
            with open(query_input_path, 'r', encoding='utf-8') as f_in, \
                open(query_output_path, 'w', encoding='utf-8') as f_out:
                    for line in tqdm(f_in, desc=f"Processing {dataset} {_split} queries"):
                        query_data = json.loads(line)
                        query = Query(id=query_data['id'], text=query_data['question'])
                        f_out.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")

            # Preprocess qrels
            qrel_input_path = os.path.join(data_dir, f"{_split}_qrels.jsonl")
            qrel_output_path = os.path.join(output_dir, f"{_split}_qrels.jsonl")
            with open(qrel_input_path, 'r', encoding='utf-8') as f_in, \
                open(qrel_output_path, 'w', encoding='utf-8') as f_out:
                    for line in tqdm(f_in, desc=f"Processing {dataset} {_split} qrels"):
                        qrel_data = json.loads(line)
                        if qrel_data["relevance"] < 1:
                            continue    # Skip non-relevant documents
                        qrel = Qrel(query_id=qrel_data['query_id'], doc_id=qrel_data['doc_id'], relevance=qrel_data['relevance'])
                        f_out.write(json.dumps(asdict(qrel), ensure_ascii=False) + "\n")
        
    print("IR Datasets preprocessing completed.")

def preprocess_msmarco_dev() -> None:
    dev_data = defaultdict(list)
    dev_path = "/home/ir_repo/work/hdd/data/dev/msmarco/top1000.dev"
    with open(dev_path, 'r') as f:
        for line in tqdm(f):
            qid, pid, query, passage = line.strip().split('\t')
            dev_data[qid].append(pid)
    output_path = "/home/ir_repo/work/hdd/data/dev/msmarco/bm25_topk_dev.json"
    with open(output_path, 'w') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)

# def preprocess_beir(overwrite: bool = False) -> None:
#     """
#     Preprocess BEIR datasets (official ZIP layout) into a simple unified format.

#     Input:
#       BASE_INPUT_DIR/beir/<ds>/
#         - queries.jsonl
#         - qrels/*.tsv  (may contain train/dev/test; we merge all)

#     Output:
#       BASE_OUTPUT_DIR/beir/<ds>/
#         - queries.jsonl   (Query: id/text)
#         - qrels.jsonl     (Qrel: query_id/doc_id/relevance)
#     """
#     print("🛠️ Preprocessing BEIR datasets...")

#     beir_in_root = os.path.join(BASE_INPUT_DIR, "beir")
#     beir_out_root = os.path.join(BASE_OUTPUT_DIR, "beir")
#     os.makedirs(beir_out_root, exist_ok=True)

#     if not os.path.exists(beir_in_root):
#         raise FileNotFoundError(f"BEIR raw folder not found: {beir_in_root}")

#     # raw/beir 하위의 dataset 폴더 자동 탐색
#     dataset_list = [
#         d for d in os.listdir(beir_in_root)
#         if os.path.isdir(os.path.join(beir_in_root, d))
#     ]
#     dataset_list.sort()

#     for ds in dataset_list:
#         in_dir = os.path.join(beir_in_root, ds)
#         out_dir = os.path.join(beir_out_root, ds)
#         os.makedirs(out_dir, exist_ok=True)

#         queries_in = os.path.join(in_dir, "queries.jsonl")
#         qrels_dir = os.path.join(in_dir, "qrels")

#         if not os.path.exists(queries_in):
#             print(f"[SKIP] {ds}: queries.jsonl not found at {queries_in}")
#             continue
#         if not os.path.exists(qrels_dir):
#             print(f"[SKIP] {ds}: qrels folder not found at {qrels_dir}")
#             continue

#         queries_out = os.path.join(out_dir, "queries.jsonl")
#         qrels_out = os.path.join(out_dir, "qrels.jsonl")

#         if (not overwrite) and os.path.exists(queries_out) and os.path.exists(qrels_out):
#             print(f"[OK] {ds}: already preprocessed, skipping.")
#             continue

#         # -------------------------
#         # 1) Queries: BEIR queries.jsonl -> {id, text}
#         #    BEIR schema: {"_id": "...", "text": "..."}
#         # -------------------------
#         print(f"  [{ds}] writing queries -> {queries_out}")
#         with open(queries_in, "r", encoding="utf-8") as f_in, \
#              open(queries_out, "w", encoding="utf-8") as f_out:
#             for line in tqdm(f_in, desc=f"{ds}: queries"):
#                 row = json.loads(line)
#                 qid = row.get("_id") if row.get("_id") is not None else row.get("id")
#                 qtext = row.get("text", "")
#                 q = Query(id=str(qid), text=qtext)
#                 f_out.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

#         # -------------------------
#         # 2) Qrels: merge all qrels/*.tsv -> {query_id, doc_id, relevance}
#         #    BEIR qrels TSV columns typically: query-id, corpus-id, score
#         #    We merge train/dev/test into ONE file.
#         # -------------------------
#         tsv_files = sorted([p for p in os.listdir(qrels_dir) if p.endswith(".tsv")])
#         if not tsv_files:
#             print(f"[SKIP] {ds}: no .tsv files in {qrels_dir}")
#             continue

#         print(f"  [{ds}] writing merged qrels -> {qrels_out}")
#         written = 0
#         with open(qrels_out, "w", encoding="utf-8") as f_out:
#             for tsv_name in tsv_files:
#                 tsv_path = os.path.join(qrels_dir, tsv_name)

#                 with open(tsv_path, "r", encoding="utf-8") as f_in:
#                     first = f_in.readline()
#                     header = first.strip().split("\t")
#                     has_header = ("query-id" in header and "corpus-id" in header)

#                     if not has_header:
#                         # 첫 줄이 데이터였으면 되돌려서 처리
#                         f_in.seek(0)

#                     for line in tqdm(f_in, desc=f"{ds}: qrels ({tsv_name})"):
#                         parts = line.rstrip("\n").split("\t")
#                         if len(parts) < 3:
#                             continue

#                         qid, did, score = parts[0], parts[1], parts[2]

#                         # score가 float인 경우도 있으니 안전 처리
#                         try:
#                             rel = int(float(score))
#                         except Exception:
#                             rel = 0

#                         # 너 기존 코드 정책 유지: relevance < 1은 skip
#                         if rel < 1:
#                             continue

#                         qrel = Qrel(query_id=str(qid), doc_id=str(did), relevance=rel)
#                         f_out.write(json.dumps(asdict(qrel), ensure_ascii=False) + "\n")
#                         written += 1

#         print(f"     [{ds}] done. qrels_written={written}, splits={tsv_files}")

#     print("BEIR preprocessing completed.")

def preprocess_beir(overwrite: bool = False) -> None:
    """
    Preprocess BEIR datasets (official ZIP layout) into unified jsonl files.

    Input:
      BASE_INPUT_DIR/beir/<ds>/
        - corpus.jsonl
        - queries.jsonl
        - qrels/*.tsv  (train/dev/test etc. -> merged)

    Output:
      BASE_OUTPUT_DIR/beir/<ds>/
        - corpus.jsonl   (Document: id/contents/title)
        - queries.jsonl  (Query: id/text)
        - qrels.jsonl    (Qrel: query_id/doc_id/relevance)
    """
    print(" Preprocessing BEIR datasets...")

    beir_in_root = os.path.join(BASE_INPUT_DIR, "beir")
    beir_out_root = os.path.join(BASE_OUTPUT_DIR, "beir")
    os.makedirs(beir_out_root, exist_ok=True)

    if not os.path.exists(beir_in_root):
        raise FileNotFoundError(f"BEIR raw folder not found: {beir_in_root}")

    dataset_list = [
        d for d in os.listdir(beir_in_root)
        if os.path.isdir(os.path.join(beir_in_root, d))
    ]
    dataset_list.sort()

    for ds in dataset_list:
        in_dir = os.path.join(beir_in_root, ds)
        out_dir = os.path.join(beir_out_root, ds)
        os.makedirs(out_dir, exist_ok=True)

        corpus_in = os.path.join(in_dir, "corpus.jsonl")
        queries_in = os.path.join(in_dir, "queries.jsonl")
        qrels_dir = os.path.join(in_dir, "qrels")

        if not os.path.exists(corpus_in):
            print(f"[SKIP] {ds}: corpus.jsonl not found at {corpus_in}")
            continue
        if not os.path.exists(queries_in):
            print(f"[SKIP] {ds}: queries.jsonl not found at {queries_in}")
            continue
        if not os.path.exists(qrels_dir):
            print(f"[SKIP] {ds}: qrels folder not found at {qrels_dir}")
            continue

        corpus_out = os.path.join(out_dir, "corpus.jsonl")
        queries_out = os.path.join(out_dir, "queries.jsonl")
        qrels_out = os.path.join(out_dir, "qrels.jsonl")

        # overwrite가 아니고 3개 다 있으면 스킵
        if (not overwrite) and all(os.path.exists(p) for p in [corpus_out, queries_out, qrels_out]):
            print(f"[OK] {ds}: already preprocessed, skipping.")
            continue

        # -------------------------
        # 1) Corpus: BEIR corpus.jsonl -> {id, contents, title}
        #    BEIR corpus schema: {"_id": "...", "title": "...", "text": "..."}
        # -------------------------
        if overwrite or (not os.path.exists(corpus_out)):
            print(f"   [{ds}] writing corpus -> {corpus_out}")
            with open(corpus_in, "r", encoding="utf-8") as f_in, \
                 open(corpus_out, "w", encoding="utf-8") as f_out:
                for line in tqdm(f_in, desc=f"{ds}: corpus"):
                    row = json.loads(line)
                    doc_id = row.get("_id") if row.get("_id") is not None else row.get("id")
                    title = row.get("title", "") or ""
                    text = row.get("text", "") or ""
                    doc = Document(id=str(doc_id), contents=text, title=title)
                    f_out.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
        else:
            print(f"  [OK] [{ds}] corpus exists, skipping.")

        # -------------------------
        # 2) Queries: BEIR queries.jsonl -> {id, text}
        #    BEIR queries schema: {"_id": "...", "text": "..."}
        # -------------------------
        if overwrite or (not os.path.exists(queries_out)):
            print(f"   [{ds}] writing queries -> {queries_out}")
            with open(queries_in, "r", encoding="utf-8") as f_in, \
                 open(queries_out, "w", encoding="utf-8") as f_out:
                for line in tqdm(f_in, desc=f"{ds}: queries"):
                    row = json.loads(line)
                    qid = row.get("_id") if row.get("_id") is not None else row.get("id")
                    qtext = row.get("text", "") or ""
                    q = Query(id=str(qid), text=qtext)
                    f_out.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
        else:
            print(f"  [OK] [{ds}] queries exists, skipping.")

        # -------------------------
        # 3) Qrels: merge all qrels/*.tsv -> {query_id, doc_id, relevance}
        #    TSV columns: query-id, corpus-id, score
        # -------------------------
        tsv_files = sorted([p for p in os.listdir(qrels_dir) if p.endswith(".tsv")])
        if not tsv_files:
            print(f"[SKIP] {ds}: no .tsv files in {qrels_dir}")
            continue

        if overwrite or (not os.path.exists(qrels_out)):
            print(f"   [{ds}] writing merged qrels -> {qrels_out}")
            written = 0
            with open(qrels_out, "w", encoding="utf-8") as f_out:
                for tsv_name in tsv_files:
                    tsv_path = os.path.join(qrels_dir, tsv_name)
                    with open(tsv_path, "r", encoding="utf-8") as f_in:
                        first = f_in.readline()
                        header = first.strip().split("\t")
                        has_header = ("query-id" in header and "corpus-id" in header)

                        if not has_header:
                            f_in.seek(0)

                        for line in tqdm(f_in, desc=f"{ds}: qrels ({tsv_name})"):
                            parts = line.rstrip("\n").split("\t")
                            if len(parts) < 3:
                                continue

                            qid, did, score = parts[0], parts[1], parts[2]
                            try:
                                rel = int(float(score))
                            except Exception:
                                rel = 0

                            if rel < 1:
                                continue

                            qrel = Qrel(query_id=str(qid), doc_id=str(did), relevance=rel)
                            f_out.write(json.dumps(asdict(qrel), ensure_ascii=False) + "\n")
                            written += 1

            print(f"    [{ds}] qrels_written={written}, merged_splits={tsv_files}")
        else:
            print(f"  [OK] [{ds}] qrels exists, skipping.")

    print("BEIR preprocessing completed.")


### ========================================================================================== ###


if __name__ == "__main__":
    exec_type = sys.argv[1]
    if exec_type == "collections":
        preprocess_collections()
    elif exec_type == "msmarco":
        preprocess_msmarco()
    elif exec_type == "ir":
        preprocess_ir_datasets()
    elif exec_type == "beir":
        preprocess_beir()
    elif exec_type == "msmarco-dev":
        preprocess_msmarco_dev()
    else:
        raise ValueError(f"Unknown exec_type: {exec_type}")