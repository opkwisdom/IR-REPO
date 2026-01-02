#!/bin/bash
CKPT_DIR=/home/ir_repo/work/hdd/checkpoint/splade/msmarco/distilbert_150k
CKPT_FILE=""
INDEX_DIR=/home/ir_repo/work/hdd/index/splade/msmarco/distilbert_150k
OUTPUT_DIR=/home/ir_repo/work/hdd/output/splade/msmarco

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 splade_train.py \
    ckpt_dir=$CKPT_DIR
echo "Training completed."

python3 indexing.py \
    ckpt_dir=$CKPT_DIR \
    ckpt_file=$CKPT_FILE \
    output_dir=$INDEX_DIR
echo "Indexing completed."

export CUDA_VISIBLE_DEVICES=0
python3 retrieval.py \
    ckpt_dir=$CKPT_DIR \
    ckpt_file=$CKPT_FILE \
    index_dir=$INDEX_DIR \
    output_dir=$OUTPUT_DIR
echo "Retrieval and evaluation completed."