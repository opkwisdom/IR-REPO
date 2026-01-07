#!/bin/bash
INDEX_DIR=/home/ir_repo/work/hdd/index/sentence-transformer/base
OUTPUT_DIR=/home/ir_repo/work/repo/sentence-transformer/results/msmarco

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 indexing.py \
    output_dir=$INDEX_DIR
echo "Indexing completed."

export CUDA_VISIBLE_DEVICES=0
python3 retrieval.py \
    index_dir=$INDEX_DIR \
    output_dir=$OUTPUT_DIR
echo "Retrieval and evaluation completed."