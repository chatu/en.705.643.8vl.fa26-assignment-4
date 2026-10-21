#!/bin/bash
# Evaluate a trained checkpoint on the held-out test groups saved by train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-models/best_model_wts.pt}
PYTHONPATH=src python -m ucf50_lrcn.run \
    --ckpt "$CKPT" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode eval
