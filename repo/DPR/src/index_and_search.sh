#!/bin/bash
CKPT_DIR=/home/ir_repo/work/hdd/checkpoint/dpr/msmarco/bert_epoch_10
# CKPT_FILE=""
INDEX_DIR=/home/ir_repo/work/hdd/index/dpr/msmarco/bert_epoch_10
OUTPUT_DIR=/home/ir_repo/work/hdd/output/dpr/msmarco

export CUDA_VISIBLE_DEVICES=0,1,2,3
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