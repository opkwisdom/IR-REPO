#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 dpr_train.py
echo "Training completed."

python3 indexing.py
echo "Indexing completed."

export CUDA_VISIBLE_DEVICES=0
python3 retrieval.py
echo "Retrieval and evaluation completed."