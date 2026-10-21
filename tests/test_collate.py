"""Regression tests for the batch collate functions."""

import torch

from ucf50_lrcn.video_datasets import collate_fn_r3d_18


def _clip(value):
    """A (T=2, C=3, H=4, W=4) clip filled with one value, so it can be identified."""
    return torch.full((2, 3, 4, 4), float(value))


def test_3d_collate_keeps_each_label_with_its_own_clip():
    """Regression: clips were filtered before labels were paired with them, so every label
    after an empty clip shifted onto the wrong video."""
    batch = [(_clip(10), 0), ([], 1), (_clip(30), 2)]
    clips, labels = collate_fn_r3d_18(batch)
    assert labels.tolist() == [0, 2]
    assert clips[:, 0, 0, 0, 0].tolist() == [10.0, 30.0]
