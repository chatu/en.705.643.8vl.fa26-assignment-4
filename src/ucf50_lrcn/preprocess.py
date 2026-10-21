"""
Module: preprocess.py

Convert UCF50 videos into directories of sampled frames, the input format VideoDataset reads.

The source tree is UCF50/<Class>/<video>.avi and the output tree mirrors it as
<frame_dir>/<Class>/<video>/frame0000.jpg ... Labels later come from the class folder names,
which is deliberate: two UCF50 classes misspell their own name inside the file names
(v_Billards_* under Billiards/, v_Pullup_* under PullUps/).

Usage:
    python -m ucf50_lrcn.preprocess --video_dir data/UCF50 --frame_dir data/UCF50_frames
"""

import argparse
import os
from multiprocessing import Pool

from tqdm import tqdm

from ucf50_lrcn.utils import get_frames, store_frames


def list_videos(video_dir):
    """
    List every video under video_dir as (class_name, video_path), sorted.

    Args:
        video_dir (str): Root folder with one subfolder of .avi files per class.

    Returns:
        list of tuple: (class_name, video_path) pairs.
    """
    videos = []
    for class_name in sorted(os.listdir(video_dir)):
        class_dir = os.path.join(video_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith('.avi'):
                videos.append((class_name, os.path.join(class_dir, name)))
    return videos


def extract_video(job):
    """
    Sample n_frames frames evenly from one video and store them as JPEGs.

    Videos whose output folder already holds n_frames frames are skipped, so an interrupted
    run can be resumed.

    Args:
        job (tuple): (video_path, out_dir, n_frames).

    Returns:
        tuple: (video_path, frames_written, frames_in_video); frames_written is -1 if skipped.
    """
    video_path, out_dir, n_frames = job
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == n_frames:
        return video_path, -1, -1
    os.makedirs(out_dir, exist_ok=True)
    frames, v_len = get_frames(video_path, n_frames=n_frames)
    store_frames(frames, out_dir)
    return video_path, len(frames), v_len


def preprocess(video_dir, frame_dir, n_frames=32, workers=os.cpu_count()):
    """
    Extract frames for every video in video_dir into frame_dir.

    Args:
        video_dir (str): UCF50 root (one subfolder per class).
        frame_dir (str): Output root; created if missing.
        n_frames (int): Frames sampled evenly from each video.
        workers (int): Parallel worker processes.

    Returns:
        list of tuple: (video_path, frames_written, frames_in_video) for videos that yielded
        fewer than n_frames frames; empty when every video is complete.
    """
    jobs = [(path, os.path.join(frame_dir, class_name, os.path.splitext(os.path.basename(path))[0]),
             n_frames) for class_name, path in list_videos(video_dir)]
    short = []
    with Pool(workers) as pool:
        for video_path, written, v_len in tqdm(pool.imap_unordered(extract_video, jobs),
                                               total=len(jobs)):
            if 0 <= written < n_frames:
                short.append((video_path, written, v_len))
    return sorted(short)


def main():
    """Parse arguments, run the extraction and report any incomplete videos."""
    parser = argparse.ArgumentParser(description='Extract UCF50 frames for training')
    parser.add_argument('--video_dir', required=True, help='UCF50 root with one folder per class')
    parser.add_argument('--frame_dir', required=True, help='Output root for frame folders')
    parser.add_argument('--n_frames', type=int, default=32, help='Frames sampled per video')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Worker processes')
    args = parser.parse_args()
    short = preprocess(args.video_dir, args.frame_dir, args.n_frames, args.workers)
    print(f'{len(short)} videos yielded fewer than {args.n_frames} frames')
    for video_path, written, v_len in short:
        print(f'  {video_path}: {written} of {args.n_frames} (video reports {v_len} frames)')


if __name__ == '__main__':
    main()
