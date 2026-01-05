# How to Use DPR Model
You can control the hyperparameters of DPR model or other options by modifying config files in `conf` folder. 

## 1. DPR Model Training
By using **Pytorch Lightning** framework, define data module and DPR model based on the `model` pre-defined in `conf/dpr_train.yaml` and train the DPR model. Since **WandbLogger** records training logs, you can check the procedure of training DPR model.

### Configuration
> conf/dpr_train.yaml
```{.yaml}
defaults:
    - data:  # Default dataset to use (msmarco)
    - model: # Default encoder model (roberta, bert)
    - train: # Training mode (base, step_based) | base: using negative sampling
num_device: The number of GPU to use
ckpt_dir: Directory to save checkpoint of the model
exp_model: Model name to record W&B dashboard 
exp_name: Individual experiment name to record W&B dashboard
```

### Executation
`$ python3 dpr_train.py`

## 2. Dense Vector Indexing
In this phase, load data and gather all vectors and ids in parallel. Specifically, by using **DDP** (Distriebuted Data Parallel), each GPU process generates context vectors and ids, then build an indexer with **FaissIndexer** library.

You can select the type of vector index by `index_key`. **flat** is a simple index, which calculates distance between two vectors using L2 (euclidean distance). On the other hand, **hnsw** uses graph-based method, which provides fast serach performance in the large dataset.

### Configuration
> conf/inexing.yaml
```{.yaml}
defaults:
    - data:   # Default dataset to use (msmarco)
    - model:  # Default encoder model (roberta, bert)
    - index:  # Retrieval algorithm (faiss, bm25)
    - search: # Search method (eval, mining) | mining: used when implementing negative sampling
index_key: # Vector indices (flat, hnsw) | hnsw: Hierarchical Navigable Small World graphs
ckpt_dir: Directory to save checkpoint of the model
output_dir: Directory to save indexer
```

### Executation
`$ python3 indexing.py`

## 3. Document Retrieval
Generate query vector, load Faiss index, and then retrieve the most relevant documents to query with Fiass indexer. Retrieved documents are stored in `output_dir`.

### Configuration
> conf/retrieval.yaml
```{.yaml}
defaults:
    - data:   # Default dataset to use (msmarco)
    - model:  # Default encoder model (roberta, bert)
    - index:  # Retrieval algorithm (faiss, bm25)
    - search: # Search method (eval, mining) | mining: used when implementing negative sampling
index_key: # Vector indices (flat, hnsw) | hnsw: Hierarchical Navigable Small World graphs
ckpt_dir:   # Directory to save checkpoint of the model
index_dir:  # Directory to use saved indices
output_dir: # Directory to save retrieved documents
```

### Executation
`$ python3 retrieval.py`

# Overall Pipeline
Above pipleline can be executable all at once by running the shell script: `all_pipline.sh`.