#!/bin/bash
# Precompute stacked optical flow at 32 evenly spaced positions of every UCF50 video into
# data/UCF50_flow, the input of the flow stream (train.sh --modality flow).
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO_DIR=${VIDEO_DIR:-data/UCF50}
FLOW_DIR=${FLOW_DIR:-data/UCF50_flow}
PYTHONPATH=src python -m ucf50_lrcn.flow \
    --video_dir "$VIDEO_DIR" \
    --flow_dir "$FLOW_DIR" \
    --n_samples 32
