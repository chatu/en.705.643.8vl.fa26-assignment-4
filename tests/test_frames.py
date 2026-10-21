"""Regression tests for frame extraction, frame storage and frame loading."""

import cv2
import numpy as np
import pytest
from torchvision import transforms

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
