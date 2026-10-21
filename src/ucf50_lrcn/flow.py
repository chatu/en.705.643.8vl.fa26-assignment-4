"""
Module: flow.py

Precompute the optical-flow input of the temporal stream. At n_samples positions spread evenly
over a video, the flow between FLOW_STACK pairs of consecutive frames is computed with
Farneback's method and stacked into 2 * FLOW_STACK channels (x0, y0, x1, y1, ...). Flow is
clipped to +-FLOW_BOUND pixels and mapped linearly onto 0..255, so zero motion is 127.5 and
reversing a motion maps a value v to 255 - v.

Each position is stored as one grayscale JPEG holding the channels tiled vertically, so the
output tree mirrors the RGB one: <flow_dir>/<Class>/<video>/flow0000.jpg ...

Usage:
    python -m ucf50_lrcn.flow --video_dir data/UCF50 --flow_dir data/UCF50_flow
"""

import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from ucf50_lrcn.preprocess import list_videos

FLOW_STACK = 5      # consecutive frame pairs per sampled position
FLOW_BOUND = 20.0   # pixels; larger displacements are clipped
# Mean and std per channel (x0, y0, x1, y1, ...) on the 0..1 scale, measured over 8 positions
# of 100 randomly chosen training-group videos: x 0.5001 / 0.0552, y 0.5020 / 0.0334.
FLOW_MEAN = [0.5001, 0.5020] * FLOW_STACK
FLOW_STD = [0.0552, 0.0334] * FLOW_STACK


def quantize_flow(flow):
    """
    Map flow in pixels to uint8, clipping at +-FLOW_BOUND.

    Args:
        flow (np.ndarray): Float flow of any shape.

    Returns:
        np.ndarray: uint8 array of the same shape.
    """
    scaled = (np.clip(flow, -FLOW_BOUND, FLOW_BOUND) + FLOW_BOUND) * 255.0 / (2 * FLOW_BOUND)
    return np.round(scaled).astype(np.uint8)


def video_flow_stacks(video_path, n_samples=32, stack=FLOW_STACK):
    """
    Compute stacked flow at n_samples positions spread evenly over a video.

    Args:
        video_path (str): Path to the video file.
        n_samples (int): Number of positions to sample.
        stack (int): Consecutive frame pairs per position.

    Returns:
        list of np.ndarray: One uint8 array of shape (2 * stack * H, W) per position, the
        channels x0, y0, x1, y1, ... tiled vertically. Empty if fewer than two frames decode.
    """
    v_cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = v_cap.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    v_cap.release()
    if len(frames) < 2:
        return []
    last = len(frames) - 1
    starts = np.linspace(0, max(last - stack, 0), n_samples).round().astype(int)
    stacks = []
    for start in starts:
        channels = []
        for k in range(stack):
            prev, nxt = frames[min(start + k, last - 1)], frames[min(start + k + 1, last)]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            channels.extend([flow[..., 0], flow[..., 1]])
        stacks.append(quantize_flow(np.concatenate(channels, axis=0)))
    return stacks


def extract_flow(job):
    """
    Compute and store the flow stacks of one video, skipping videos already complete.

    Args:
        job (tuple): (video_path, out_dir, n_samples).

    Returns:
        tuple: (video_path, stacks_written); stacks_written is -1 if skipped.
    """
    video_path, out_dir, n_samples = job
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == n_samples:
        return video_path, -1
    os.makedirs(out_dir, exist_ok=True)
    stacks = video_flow_stacks(video_path, n_samples)
    for idx, tiled in enumerate(stacks):
        cv2.imwrite(os.path.join(out_dir, f'flow{idx:04d}.jpg'), tiled,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return video_path, len(stacks)


def preprocess_flow(video_dir, flow_dir, n_samples=32, workers=os.cpu_count()):
    """
    Compute flow stacks for every video in video_dir into flow_dir.

    Args:
        video_dir (str): UCF50 root (one subfolder per class).
        flow_dir (str): Output root; created if missing.
        n_samples (int): Positions sampled evenly from each video.
        workers (int): Parallel worker processes.

    Returns:
        list of tuple: (video_path, stacks_written) for videos that yielded fewer than
        n_samples stacks; empty when every video is complete.
    """
    jobs = [(path, os.path.join(flow_dir, class_name, os.path.splitext(os.path.basename(path))[0]),
             n_samples) for class_name, path in list_videos(video_dir)]
    short = []
    with Pool(workers) as pool:
        for video_path, written in tqdm(pool.imap_unordered(extract_flow, jobs), total=len(jobs)):
            if 0 <= written < n_samples:
                short.append((video_path, written))
    return sorted(short)


def main():
    """Parse arguments, run the flow extraction and report any incomplete videos."""
    parser = argparse.ArgumentParser(description='Precompute stacked optical flow for UCF50')
    parser.add_argument('--video_dir', required=True, help='UCF50 root with one folder per class')
    parser.add_argument('--flow_dir', required=True, help='Output root for flow folders')
    parser.add_argument('--n_samples', type=int, default=32, help='Positions sampled per video')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Worker processes')
    args = parser.parse_args()
    short = preprocess_flow(args.video_dir, args.flow_dir, args.n_samples, args.workers)
    print(f'{len(short)} videos yielded fewer than {args.n_samples} flow stacks')
    for video_path, written in short:
        print(f'  {video_path}: {written}')


if __name__ == '__main__':
    main()
