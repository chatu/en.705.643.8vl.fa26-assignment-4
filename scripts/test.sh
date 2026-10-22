#!/bin/bash
# Evaluate a trained checkpoint on the held-out test groups saved by train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_DIR=${OUT_DIR:-runs/rgb}
CKPT=${CKPT:-$OUT_DIR/models/best_model_wts.pt}
PYTHONPATH=src python -m ucf50_lrcn.run \
    --ckpt "$CKPT" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --num_workers "${NUM_WORKERS:-0}" \
    --out_dir "$OUT_DIR" \
    --mode eval "$@"
