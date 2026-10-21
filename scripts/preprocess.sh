#!/bin/bash
# Extract 32 evenly spaced frames from every UCF50 video into data/UCF50_frames.
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO_DIR=${VIDEO_DIR:-data/UCF50}
FRAME_DIR=${FRAME_DIR:-data/UCF50_frames}
PYTHONPATH=src python -m ucf50_lrcn.preprocess \
    --video_dir "$VIDEO_DIR" \
    --frame_dir "$FRAME_DIR" \
    --n_frames 32
