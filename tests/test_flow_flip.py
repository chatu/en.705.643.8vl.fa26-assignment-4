"""Regression test for flipping flow clips."""

import torch

from ucf50_lrcn.utils import RandomFlowFlip


def test_flipping_a_flow_clip_reverses_horizontal_motion_only():
    """A mirrored video moves the other way horizontally and the same way vertically, so the
    flip must reflect the x channels about zero motion and leave the y channels alone."""
    clip = torch.zeros(2, 4, 3, 5, dtype=torch.uint8)
    clip[:, 0::2] = 150                       # x channels: rightward motion
    clip[:, 1::2] = 100                       # y channels: upward motion
    clip[:, 1::2, :, 0] = 90                  # a mark in the left column of the y channels
    flipped = RandomFlowFlip(p=1.0)(clip.clone())
    assert torch.all(flipped[:, 0::2] == 105)            # 255 - 150: leftward motion
    assert torch.all(flipped[:, 1::2, :, :-1] == 100)    # vertical motion unchanged ...
    assert torch.all(flipped[:, 1::2, :, -1] == 90)      # ... and mirrored with the image
