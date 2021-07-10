#!/bin/bash

python scholar/run_scholar.py \
    ./data/20ng/processed-dev \
    --dev-metric npmi \
    -k 50 \
    --epochs 500 \
    --patience 500 \
    --batch-size 200 \
    --background-embeddings \
    --device 0 \
    --dev-prefix dev \
    -lr 0.002 \
    --alpha 0.5 \
    --eta-bn-anneal-step-const 0.25 \
    --doc-reps-dir ./data/20ng/processed-dev/logits/checkpoint-9000/doc_logits \
    --use-doc-layer \
    --no-bow-reconstruction-loss \
    --doc-reconstruction-weight 0.5 \
    --doc-reconstruction-temp 1.0 \
    --doc-reconstruction-logit-clipping 10.0 \
    -o ./outputs/20ng