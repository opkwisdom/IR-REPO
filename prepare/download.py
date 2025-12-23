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

from resources import MSMARCO_MAP, HF_MAP, WIKI_DUMP, IR_MAP, BEIR_MAP

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


def download_beir_datasets() -> None:
    """
    Download BEIR benchmark datasets via ir-datasets (NQ, etc.).
    BEIR benchmark is the industry standard for evaluating zero-shot retrieval models.
    """
    raise NotImplementedError("BEIR dataset download not implemented yet.")



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