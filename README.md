# Neural IR Framework (w/ Hydra)
A unified framework for training and evaluating Neural IR models. This repository supports end-to-end pipelines from preprocessing to training and evluation on major benchmarks.

# Environmental Setup
## 🐋 Docker
First, pull the Docker image and set permissions:
```{.bash}
docker pull juno2357/ir_repo:latest
sudo chmod -R 777 <your_hdd_mount_point>    # Give full permissions
```
There are two ways to run the container:
1. Using VSCode DevContainer
    - Make .devcontainer directory on the project root and add the `devcontainer.json` file, following the template (`ex_devcontainer.json`)
    - Open Folder in DevContainer via VSCode.
2. Using Docker CLI
```{.bash}
docker run -it --gpus all \
    --shm-size 64g \
    --name <your_container_name> \
    -v $(pwd):/home/ir_repo/work \
    -v <your_hdd_mount_point>:/home/ir_repo/work/hdd \
    juno2357/ir_repo:latest \
    /bin/bash   # or zsh
```

## 🐍 Conda
Execute the following commands to set up the Conda environment:
```{.bash}
conda env create -f ir_repo_env.yml
conda activate ir_repo
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.57.1
pip install pytorch-lightning==2.5.6
```

# Results

## MS Marco Passage Retrieval (Ours)
* These reported DPR performance is trained using only MSMARCO official triples

| Model | MRR@10 | nDCG@10 | Recall@10 | Recall@1000 |
| :--- | :--- | :--- | :--- | :--- |
| BM25 (Tuned) | 0.1874 | 0.2340 | 0.3916 | 0.8573 |
| DPR (Bert) | 0.2677 | 0.3202 | 0.4975 | 0.9313 |
| DPR (Roberta) | 0.3011 | 0.3595 | 0.5578 | 0.9520 |
| SPLADE-max (Distilbert) | 0.3417 | 0.4048 | 0.6160 | 0.9641 |
| ColBERT_v1 | - | - | - | - |
| ColBERT_v2 | - | - | - | - |
| S-BERT (Bert) | 0.3809 | 0.4440 | 0.6577 | 0.9777 |