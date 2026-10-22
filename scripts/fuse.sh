#!/bin/bash
# Late fusion: average the softmax of a trained RGB model and a trained flow model and report
# each stream's and the fused accuracy on the validation and test groups.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m ucf50_lrcn.fuse \
    --rgb_dir "${FRAME_DIR:-data/UCF50_frames}" \
    --rgb_ckpt "${RGB_CKPT:-runs/rgb/models/best_model_wts.pt}" \
    --flow_dir "${FLOW_DIR:-data/UCF50_flow}" \
    --flow_ckpt "${FLOW_CKPT:-runs/flow/models/best_model_wts.pt}" \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 8 \
    --num_workers "${NUM_WORKERS:-0}" "$@"
