"""Regression tests for the optical-flow input."""

import cv2
import numpy as np

from ucf50_lrcn.flow import FLOW_BOUND, FLOW_STACK, video_flow_stacks


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
    """Channels are tiled x0, y0, x1, y1, ...; the horizontal flip negates the even ones,
    so a swap here would make the flip reverse vertical motion instead."""
    _write_rightward_video(tmp_path / "v.avi")
    stacks = video_flow_stacks(str(tmp_path / "v.avi"), n_samples=4)
    assert len(stacks) == 4
    channels = stacks[0].reshape(2 * FLOW_STACK, 64, 96)[:, 16:48, 24:72].astype(float)
    pixels = (channels * 2 * FLOW_BOUND / 255.0) - FLOW_BOUND
    x_motion, y_motion = np.median(pixels[0::2], axis=(1, 2)), np.median(pixels[1::2], axis=(1, 2))
    assert np.all(np.abs(x_motion - 2.0) < 0.5), x_motion
    assert np.all(np.abs(y_motion) < 0.5), y_motion
