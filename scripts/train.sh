#!/bin/bash
# Train the LRCN on UCF50 with the group split (val groups 19-21, test groups 22-25).
set -euo pipefail
cd "$(dirname "$0")/.."
FRAME_DIR=${FRAME_DIR:-data/UCF50_frames}
PYTHONPATH=src python -m ucf50_lrcn.run \
    --frame_dir "$FRAME_DIR" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode train
