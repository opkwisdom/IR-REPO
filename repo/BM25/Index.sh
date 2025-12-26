# !/bin/bash

# Base index dir
INDEX_DIR=/home/ir_repo/work/hdd/index/bm25
INPUT_DIR=/home/ir_repo/work/hdd/data/preprocessed

Index() {
  INPUT="$INPUT_DIR"/"$1"
  INDEX="$INDEX_DIR"/"$2"

  echo "Indexing collection from $INPUT to $INDEX"
  python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input $INPUT \
    --index $INDEX \
    --generator DefaultLuceneDocumentGenerator \
    --threads 64 \
    --storePositions --storeDocvectors --storeRaw
  echo "Indexing completed."
}

# k1=0.82, b=0.68 are tuned on MSMARCO passage ranking dev set

Index msmarco_collection msmarco
Index psgs_w100_collection psgs_w100
