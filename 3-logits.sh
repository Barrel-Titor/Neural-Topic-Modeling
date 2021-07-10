#!/bin/bash

python -m torch.distributed.launch \
	--nproc_per_node 1 \
	teacher/bert_reconstruction.py \
  --output-dir ./data/20ng/processed-dev/logits \
  --seed 42 \
  --num-workers 6 \
  --get-reps \
  --checkpoint-folder-pattern "checkpoint-9000" \
  --save-doc-logits \
  --no-dev