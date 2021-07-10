#!/bin/bash

#cd ..

source activate teacher

python -m torch.distributed.launch \
  --nproc_per_node 4 \
	teacher/bert_reconstruction.py \
  --input-dir='./data/20ng/processed-dev' \
  --output-dir='./data/20ng/processed-dev/logits' \
  --do-train \
  --evaluate-during-training \
  --logging-steps 200 \
  --save-steps 500 \
  --num-train-epochs 50 \
  --seed 42 \
  --num-workers 4 \
  --batch-size 20 \
  --gradient-accumulation-steps 8

python -m torch.distributed.launch \
  --nproc_per_node 4 \
  multi_gpu.py