"""Tests for the optical-flow input: how stored flow reads back, and the flip."""

import cv2
import numpy as np

from ucf50_lrcn.flow import FLOW_BOUND, FLOW_STACK, extract_flow
from ucf50_lrcn.video_datasets import VideoDataset


def _write_rightward_video(path, n_frames=20, shift=2):
    """An .avi of a smooth random texture moving `shift` pixels to the right per frame."""
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (64, 160)).astype(np.uint8), (0, 0), 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (96, 64))
    for i in range(n_frames):
        frame = texture[:, 48 - shift * i: 48 - shift * i + 96]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def test_flow_channels_alternate_x_then_y(tmp_path):
    """Channels are x0, y0, x1, y1, ... once stored and read back by the dataset; the
    horizontal flip negates the even ones, so a swap or a wrong unpacking here would make
    the flip reverse vertical motion instead."""
    _write_rightward_video(tmp_path / "v.avi")
    extract_flow((str(tmp_path / "v.avi"), str(tmp_path / "flow"), 4))
    clip, _ = VideoDataset([(str(tmp_path / "flow"), 0)], fr_per_vid=4,
                           flow_channels=2 * FLOW_STACK)[0]
    assert tuple(clip.shape) == (4, 2 * FLOW_STACK, 64, 96)
    pixels = clip[0, :, 16:48, 24:72].double() * 2 * FLOW_BOUND / 255.0 - FLOW_BOUND
    x_motion = pixels[0::2].flatten(1).median(dim=1).values
    y_motion = pixels[1::2].flatten(1).median(dim=1).values
    assert bool(((x_motion - 2.0).abs() < 0.5).all()), x_motion
    assert bool((y_motion.abs() < 0.5).all()), y_motion
