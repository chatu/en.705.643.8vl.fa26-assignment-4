"""Regression tests for frame extraction, frame storage and frame loading."""

import cv2
import numpy as np
import pytest

from ucf50_lrcn.utils import get_frames, store_frames
from ucf50_lrcn.video_datasets import VideoDataset

N_SOURCE_FRAMES = 40


def _frame_number(frame):
    """Recover the frame number a synthetic frame encodes in its brightness."""
    return int(round(float(np.mean(frame)) / 5.0))


@pytest.fixture(name="numbered_video")
def fixture_numbered_video(tmp_path):
    """An .avi whose frame i is a flat grey image of brightness 5 * i."""
    path = tmp_path / "numbered.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 32))
    for i in range(N_SOURCE_FRAMES):
        writer.write(np.full((32, 32, 3), 5 * i, dtype=np.uint8))
    writer.release()
    return str(path)


def test_get_frames_returns_exactly_the_number_requested(numbered_video):
    """Regression: np.linspace(..., n_frames + 1) sampled one frame too many."""
    frames, n_total = get_frames(numbered_video, n_frames=16)
    assert n_total == N_SOURCE_FRAMES
    assert len(frames) == 16


def test_stored_frames_sort_in_temporal_order(tmp_path):
    """Regression: names like frame10.jpg sorted before frame2.jpg."""
    frames = [np.full((8, 8, 3), 5 * i, dtype=np.uint8) for i in range(12)]
    store_frames(frames, str(tmp_path))
    names = sorted(p.name for p in tmp_path.glob("*.jpg"))
    numbers = [_frame_number(cv2.imread(str(tmp_path / n))) for n in names]
    assert numbers == list(range(12))


def _frame_dir(tmp_path, n_frames):
    """A directory of stored, numbered frames, as preprocessing would produce."""
    store_frames([np.full((8, 8, 3), 5 * i, dtype=np.uint8) for i in range(n_frames)],
                 str(tmp_path))
    return str(tmp_path)


def test_dataset_loads_frames_in_temporal_order(tmp_path):
    """Regression: glob.glob returns files in arbitrary order, scrambling time."""
    dataset = VideoDataset([(_frame_dir(tmp_path, 12), 0)], fr_per_vid=12)
    frames, _ = dataset[0]
    numbers = [_frame_number(f.numpy()) for f in frames]
    assert numbers == list(range(12))


def test_dataset_samples_frames_across_the_whole_clip(tmp_path):
    """Regression: fr_paths[:fr_per_vid] kept only the start of the clip."""
    dataset = VideoDataset([(_frame_dir(tmp_path, 32), 0)], fr_per_vid=8)
    frames, _ = dataset[0]
    numbers = [_frame_number(f.numpy()) for f in frames]
    assert len(numbers) == 8
    assert numbers[0] == 0
    assert numbers[-1] == 31
