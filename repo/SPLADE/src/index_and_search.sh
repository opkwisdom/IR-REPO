#!/bin/bash
CKPT_DIR=/home/ir_repo/work/hdd/checkpoint/splade/msmarco/distilbert_150k_flops
# CKPT_FILE=""
INDEX_DIR=/home/ir_repo/work/hdd/index/splade/msmarco/distilbert_150k_flops
OUTPUT_DIR=/home/ir_repo/work/hdd/output/splade/msmarco

python3 indexing.py \
    ckpt_dir=$CKPT_DIR \
    output_dir=$INDEX_DIR
echo "Indexing completed."

export CUDA_VISIBLE_DEVICES=0
python3 retrieval.py \
    ckpt_dir=$CKPT_DIR \
    index_dir=$INDEX_DIR \
    output_dir=$OUTPUT_DIR
echo "Retrieval and evaluation completed."