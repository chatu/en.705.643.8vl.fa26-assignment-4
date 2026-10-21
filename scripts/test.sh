#!/bin/bash
PYTHONPATH=src python -m ucf50_lrcn.run \
    --ckpt /home/vince/HMDB_Vid_Classification/models/epoch4_model_wts.pt \
    --model_type lrcn \
    --n_classes 51 \
    --model_type lrcn \
    --batch_size 4 \
    --mode eval
