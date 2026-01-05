import os
import requests
from tqdm import tqdm
import ir_datasets
import subprocess
import tarfile
import json
import numpy as np
from datasets import load_dataset
import sys
import zipfile


from resources import MSMARCO_MAP, HF_MAP, WIKI_DUMP, IR_MAP, BEIR_MAP, BEIR_ZIP_MAP

BASE_DATA_DIR = "/home/ir_repo/work/hdd/data/raw"

def download_wiki_dump() -> None:
    wiki_url = WIKI_DUMP
    print("\n📦 Downloading Wikipedia dumps...")

    dest_dir = os.path.join(BASE_DATA_DIR, "collection")
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", "-P", dest_dir, wiki_url], check=True)
    print(f"Wikipedia dump downloaded to {dest_dir}")


# def download_msmarco() -> None:
#     """Download MSMARCO benchmark datasets."""
#     print("\n📦 Downloading MSMARCO datasets...")

#     for ds_key in MSMARCO_MAP.keys():
#         dest_dir = os.path.join(BASE_DATA_DIR, ds_key)
#         os.makedirs(dest_dir, exist_ok=True)

#         dataset = MSMARCO_MAP[ds_key]
#         for name, url in dataset.items():
#             output_path = os.path.join(dest_dir, name)

#             # 이미 존재하는 경우 스킵
#             if os.path.exists(output_path) or (
#                 os.path.exists(output_path.rstrip(".tar.gz"))
#             ):
#                 print(f"{name} already exists, skipping.")
#                 continue

#             subprocess.run(["wget", "-O", output_path, url], check=True)
#             print(f"{ds_key} {name} saved to {output_path}")
            
#             # 압축 파일이면 자동 해제
#             if output_path.endswith(".tar.gz"):
#                 extract_dir = os.path.dirname(output_path)
#                 print(f"📦 Extracting tar.gz to {extract_dir} ...")
#                 try:
#                     with tarfile.open(output_path, "r:gz") as tar:
#                         tar.extractall(path=extract_dir)
#                     print(f"Extracted to {extract_dir}")
#                 except Exception as e:
#                     print(f"Extraction failed: {e}")
#                 else:
#                     os.remove(output_path)
#                     print(f"Removed archive: {output_path}")


def download_hf_datasets() -> None:
    """Download IR benchmark datasets via HF (NQ, TriviaQA, etc.)."""
    print("\n📦 Downloading datasets via 🤗 Hugging Face Datasets...")
    base_data_dir = os.path.join(BASE_DATA_DIR, "hf")
    os.makedirs(base_data_dir, exist_ok=True)
    
    for ds_key in HF_MAP.keys():
        dest_dir = os.path.join(base_data_dir, ds_key)
        os.makedirs(dest_dir, exist_ok=True)

        dataset = HF_MAP[ds_key]
        path = dataset["path"]
        name = dataset["name"]

        # 이미 존재하는 경우 스킵
        if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
            print(f"{name} already exists, skipping.")
            continue

        def safe_convert(example):
            ans = example["answer"]
            if isinstance(ans, np.ndarray):
                example["answer"] = ans.tolist()
            elif isinstance(ans, list) and len(ans) and isinstance(ans[0], np.ndarray):
                example["answer"] = [a.tolist() for a in ans]
            return example
        
        def save_to_jsonl(output_path, examples):
            with open(output_path, 'w', encoding="utf-8") as f:
                for row in tqdm(examples):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        try:
            ds = load_dataset(path, name) if name is not None else load_dataset(path)
            for split in ["train", "validation", "test"]:
                if split in ds:
                    output_path = os.path.join(dest_dir, f"{split}.jsonl")
                    ds_split = ds[split].map(safe_convert)
                    save_to_jsonl(output_path, ds_split)
                    print(f"Saved {split} -> {output_path}")
            print(f"{ds_key} downloaded to {dest_dir}")
        except Exception as e:
            print(f"Failed to download {ds_key}: {e}")


def download_ir_datasets(overwrite: bool = False) -> None:
    """
    Download standard IR benchmark datasets (e.g., NQ, TriviaQA, MS MARCO) via `ir_datasets`.
    These datasets are widely used for training and evaluating retrieval models.
    """
    print("\n📦 Downloading datasets via IR Datasets...")
    
    ir_resources = ["train", "dev"]
    base_data_dir = os.path.join(BASE_DATA_DIR, "datasets")
    os.makedirs(base_data_dir, exist_ok=True)
    
    for ds_key in IR_MAP.keys():
        dataset_conf = IR_MAP[ds_key]
        output_dir = os.path.join(base_data_dir, ds_key)
        if os.path.exists(output_dir) and not overwrite:
            print(f"{ds_key} already exists, skipping.")
            continue
        os.makedirs(output_dir, exist_ok=True)
        for ir_resource in ir_resources:
            try:
                dataset_name = dataset_conf[ir_resource]
                dataset = ir_datasets.load(dataset_name)
                if ir_resource == "passages":
                    dest_path = os.path.join(BASE_DATA_DIR, "collection", f"{ds_key}_{ir_resource}.jsonl")
                    if os.path.exists(dest_path) and not overwrite:
                        print(f"{ds_key} {ir_resource} queries already exists, skipping.")
                        continue
                    # iterate and save passages
                    with open(dest_path, 'w', encoding="utf-8") as f:
                        for doc in tqdm(dataset.docs_iter(), desc=f"Downloading {ds_key} {ir_resource}"):
                            json_line = {
                                "id": doc.doc_id,
                                "text": doc.text,
                                "title": doc.document_title
                            }
                            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                else:
                    dest_path = os.path.join(base_data_dir, ds_key, f"{ir_resource}_queries.jsonl")
                    if os.path.exists(dest_path) and not overwrite:
                        print(f"{ds_key} {ir_resource} queries already exists, skipping.")
                        continue
                    # iterate and save queries
                    with open(dest_path, 'w', encoding="utf-8") as f:
                        for query in tqdm(dataset.queries_iter(), desc=f"Downloading {ds_key} {ir_resource}"):
                            json_line = {
                                "id": query.query_id,
                                "question": query.text
                            }
                            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

                    # iterate and save qrels
                    dest_path = os.path.join(base_data_dir, ds_key, f"{ir_resource}_qrels.jsonl")
                    if os.path.exists(dest_path) and not overwrite:
                        print(f"{ds_key} {ir_resource} queries already exists, skipping.")
                        continue
                    with open(dest_path, 'w', encoding="utf-8") as f:
                        for qrel in tqdm(dataset.qrels_iter(), desc=f"Downloading {ds_key} {ir_resource} qrels"):
                            json_line = {
                                "query_id": qrel.query_id,
                                "doc_id": qrel.doc_id,
                                "relevance": qrel.relevance
                            }
                            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to download {ds_key} - {ir_resource}: {e}")
                continue


# def download_beir_datasets() -> None:
#     """
#     Download BEIR benchmark datasets via ir-datasets (NQ, etc.).
#     BEIR benchmark is the industry standard for evaluating zero-shot retrieval models.
#     """
#     raise NotImplementedError("BEIR dataset download not implemented yet.")
def download_beir_datasets(overwrite: bool = False, remove_zip: bool = True) -> None:
    """
    Download BEIR benchmark datasets from the official BEIR ZIP distribution.
    Each dataset is stored under: {BASE_DATA_DIR}/beir/<dataset>/
      - corpus.jsonl
      - queries.jsonl
      - qrels/*.tsv
    """
    print("\n📦 Downloading BEIR datasets (official ZIP)...")

    base_beir_dir = os.path.join(BASE_DATA_DIR, "beir")
    os.makedirs(base_beir_dir, exist_ok=True)

    for ds_key, url in BEIR_ZIP_MAP.items():
        if url is None:
            print(f"[SKIP] {ds_key}: URL is None (not publicly available / not configured).")
            continue

        ds_dir = os.path.join(base_beir_dir, ds_key)
        os.makedirs(ds_dir, exist_ok=True)

        # Heuristic "done" check: corpus.jsonl & queries.jsonl 존재하면 완료로 간주
        corpus_path = os.path.join(ds_dir, "corpus.jsonl")
        queries_path = os.path.join(ds_dir, "queries.jsonl")

        if not overwrite and os.path.exists(corpus_path) and os.path.exists(queries_path):
            print(f"[OK] {ds_key} already exists, skipping.")
            continue

        zip_name = f"{ds_key}.zip"
        zip_path = os.path.join(ds_dir, zip_name)

        # 1) Download ZIP
        if overwrite and os.path.exists(zip_path):
            os.remove(zip_path)

        if not os.path.exists(zip_path):
            print(f"⬇️  Downloading {ds_key} -> {zip_path}")
            subprocess.run(["wget", "-O", zip_path, url], check=True)
        else:
            print(f"[CACHE] ZIP already downloaded: {zip_path}")

        # 2) Extract ZIP
        print(f"📦 Extracting {zip_name} -> {ds_dir}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(ds_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Bad zip file for {ds_key}: {zip_path}") from e

        # 3) Post-check: basic expected files
        if not (os.path.exists(corpus_path) and os.path.exists(queries_path)):
            # 일부 zip은 내부에 ds_key/ 폴더를 한 겹 더 둘 수도 있어서 fallback 체크
            nested_dir = os.path.join(ds_dir, ds_key)
            nested_corpus = os.path.join(nested_dir, "corpus.jsonl")
            nested_queries = os.path.join(nested_dir, "queries.jsonl")
            nested_qrels = os.path.join(nested_dir, "qrels")

            if os.path.exists(nested_corpus) and os.path.exists(nested_queries):
                # nested 구조면 파일들을 상위로 “정리”해준다 (파이프라인 단순화용)
                print(f"🧹 Detected nested folder structure for {ds_key}. Flattening...")
                # move corpus/queries
                os.replace(nested_corpus, corpus_path)
                os.replace(nested_queries, queries_path)
                # move qrels folder
                if os.path.exists(nested_qrels):
                    target_qrels = os.path.join(ds_dir, "qrels")
                    if os.path.exists(target_qrels):
                        # 기존 qrels 있으면 overwrite 방식으로 정리
                        # (여기서는 단순히 nested 쪽 파일을 덮어쓴다고 가정)
                        pass
                    os.makedirs(target_qrels, exist_ok=True)
                    for fn in os.listdir(nested_qrels):
                        os.replace(os.path.join(nested_qrels, fn), os.path.join(target_qrels, fn))
                # remove nested dir if empty-ish
                try:
                    # nested_dir 안에 남은 게 없으면 삭제
                    for root, dirs, files in os.walk(nested_dir, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(nested_dir)
                except Exception:
                    pass
            else:
                raise RuntimeError(
                    f"{ds_key} extracted, but corpus.jsonl/queries.jsonl not found in {ds_dir}. "
                    f"Please inspect extracted contents."
                )

        # 4) Optionally remove ZIP
        if remove_zip and os.path.exists(zip_path):
            os.remove(zip_path)

        print(f"✅ {ds_key} downloaded & extracted to {ds_dir}")



if __name__ == "__main__":
    exec_type = sys.argv[1]
    # if exec_type == "msmarco":
    #     download_msmarco()
    if exec_type == "hf":
        download_hf_datasets()
    elif exec_type == "wiki":
        download_wiki_dump()
    elif exec_type == "ir":
        download_ir_datasets()
    elif exec_type == "beir":
        download_beir_datasets()
    else:
        raise ValueError(f"Unknown execution type: {exec_type}")