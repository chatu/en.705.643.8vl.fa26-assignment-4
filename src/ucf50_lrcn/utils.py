"""
Module: utils.py

This module provides helper functions for video processing and data transformations
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets, for RGB clips and,
      with RandomFlowFlip, for optical-flow clips.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
from ucf50_lrcn.video_datasets import collate_fn_r3d_18, collate_fn_rnn


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Number of frames that decode (not the header's frame count).

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - Frames are sampled at uniformly spaced indices.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    # Count the frames that actually decode. CAP_PROP_FRAME_COUNT comes from the container
    # header, which overstates the real count for some UCF50 files (v_GolfSwing_g22_c03 claims
    # 348 frames and decodes 276), so indices computed from it can point past the end and the
    # clip silently loses its ending.
    v_len = 0
    while v_cap.grab():
        v_len += 1
    v_cap.release()
    if v_len <= 0:
        print("No frames found in video:", vid)
        return frames, 0
    frame_idx = set(np.linspace(0, v_len - 1, n_frames).round().astype(int).tolist())
    v_cap = cv2.VideoCapture(vid)
    for idx in range(v_len):
        if not v_cap.grab():
            break
        if idx in frame_idx:
            success, frame = v_cap.retrieve()
            if success:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    v_cap.release()
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Zero-padded so that a plain sort of the names is also temporal order
        # (unpadded, frame10.jpg sorted before frame2.jpg).
        path_to_frame = os.path.join(store_path, f'frame{idx:04d}.jpg')
        cv2.imwrite(path_to_frame, frame)


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std):
    """
    Compose and return data transforms for training and validation/test datasets.

    Both operate on a whole clip, a uint8 tensor of shape (T, C, H, W). The training transforms
    add a random horizontal flip and a random translation, drawn once per clip and shared by
    all its frames; the validation/test transforms only resize, rescale to [0, 1] and normalize.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    # torchvision.transforms.v2 transforms a (T, C, H, W) clip tensor as one image batch:
    # each random transform draws its parameters once and applies them to every frame.
    train_transforms = transforms.Compose([
        transforms.Resize((height, width), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


class RandomFlowFlip(torch.nn.Module):
    """
    Horizontally flip a uint8 flow clip of shape (T, C, H, W) with probability p.

    Mirroring the image reverses horizontal motion, so the x channels (even indices, from
    ucf50_lrcn.flow) are also reflected about zero motion, v -> 255 - v. A plain image flip
    would leave every horizontal motion pointing the wrong way.

    Args:
        p (float): Probability of flipping.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, clip):
        """Flip the whole clip, or return it unchanged."""
        if torch.rand(1).item() >= self.p:
            return clip
        clip = clip.flip(-1)
        clip[:, 0::2] = 255 - clip[:, 0::2]
        return clip


def compose_flow_transforms(height, width, mean, std):
    """
    Compose and return data transforms for flow clips, the counterpart of compose_data_transforms.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean per flow channel, on the 0..1 scale.
        std (list): Standard deviation per flow channel, on the 0..1 scale.

    Returns:
        tuple: (train_transforms, val_test_transforms)
    """
    train_transforms = transforms.Compose([
        transforms.Resize((height, width), antialias=True),
        RandomFlowFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=128),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((height, width), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn', num_workers=0):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Worker processes that load clips in parallel with training.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    if model == "lrcn":
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_rnn,
                             num_workers=num_workers)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_rnn,
                             num_workers=num_workers)
    else:
        train_dl = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn_r3d_18,
                             num_workers=num_workers)
        val_dl = DataLoader(val_dataset, batch_size=2 * batch_size,
                            shuffle=False, collate_fn=collate_fn_r3d_18,
                             num_workers=num_workers)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn', num_workers=0):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Worker processes that load clips in parallel with evaluation.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    if model == "lrcn":
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_rnn,
                             num_workers=num_workers)
    else:
        test_dl = DataLoader(test_dataset, batch_size=2 * batch_size,
                             shuffle=False, collate_fn=collate_fn_r3d_18,
                             num_workers=num_workers)
    dataloaders = {'test': test_dl}
    return dataloaders
