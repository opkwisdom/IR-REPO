import json
import os
from tqdm import tqdm
import glob
import sys
from dataclasses import dataclass, asdict

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
### ========================================================================================== ###


if __name__ == "__main__":
    exec_type = sys.argv[1]
    if exec_type == "collections":
        preprocess_collections()
    elif exec_type == "msmarco":
        preprocess_msmarco()
    elif exec_type == "ir":
        preprocess_ir_datasets()
    else:
        raise ValueError(f"Unknown exec_type: {exec_type}")