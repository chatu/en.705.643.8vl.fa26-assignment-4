"""Regression tests for data augmentation."""

import numpy as np
import torch
from PIL import Image

from ucf50_lrcn.utils import compose_data_transforms, transform_stats
from ucf50_lrcn.video_datasets import VideoDataset


def _clip_of_identical_frames(tmp_path, n_frames=8):
    """A frame directory whose frames are all the same asymmetric image."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :20] = 255          # a bright band on the left only, so a flip is visible
    image[10:20, 30:40] = 128    # an off-centre block, so a translation is visible
    for i in range(n_frames):
        Image.fromarray(image).save(tmp_path / f"frame{i:04d}.jpg", quality=100)
    return str(tmp_path)


def test_training_augmentation_is_the_same_for_every_frame_of_a_clip(tmp_path):
    """Regression: random flips and translations were drawn per frame, so frames of one clip
    were flipped and shifted inconsistently, which destroys the motion the LSTM models."""
    h, w, mean, std = transform_stats('lrcn')
    train_transforms, _ = compose_data_transforms(h, w, mean, std)
    dataset = VideoDataset([(_clip_of_identical_frames(tmp_path), 0)], fr_per_vid=8,
                           transforms=train_transforms)
    for seed in range(20):
        torch.manual_seed(seed)
        frames, _ = dataset[0]
        for frame in frames[1:]:
            torch.testing.assert_close(frame, frames[0])


def test_clip_tensor_has_the_model_input_shape(tmp_path):
    """A clip comes out as (time, channels, height, width) at the model's input size."""
    h, w, mean, std = transform_stats('lrcn')
    _, val_transforms = compose_data_transforms(h, w, mean, std)
    dataset = VideoDataset([(_clip_of_identical_frames(tmp_path), 0)], fr_per_vid=8,
                           transforms=val_transforms)
    # pylint: disable=no-member  # pylint infers the [] returned for an empty clip
    frames, label = dataset[0]
    assert tuple(frames.shape) == (8, 3, h, w)
    assert frames.dtype == torch.float32
    assert label == 0
