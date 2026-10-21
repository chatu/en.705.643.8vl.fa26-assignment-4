"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

import os
import re
import glob
from tqdm import tqdm
import numpy as np


def frame_index(frame_path):
    """
    Return the frame number encoded in a frame file name such as frame0012.jpg.

    Sorting on this number, rather than on the name as text, keeps temporal order for both
    zero-padded and unpadded names.

    Args:
        frame_path (str): Path to a frame image.

    Returns:
        int: The frame number.
    """
    return int(re.search(r'(\d+)\.jpg$', frame_path).group(1))


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory, in temporal order (glob returns
        # files in arbitrary order), and select up to fr_per_vid frames
        fr_paths = sorted(glob.glob(self.dataset[idx][0] + '/*.jpg'), key=frame_index)
        # Sample fr_per_vid frames evenly from the first to the last, so the clip covers the
        # whole action rather than only its opening frames.
        if len(fr_paths) > self.fpv:
            keep = np.linspace(0, len(fr_paths) - 1, self.fpv).round().astype(int)
            fr_paths = [fr_paths[i] for i in keep]
        
        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        
        # Get the label associated with the video
        fr_label = self.dataset[idx][1]
        
        # Apply transforms to each frame if provided, else keep original images
        fr_imgs_trans = [self.transforms(fr_img) for fr_img in fr_imgs] if self.transforms else fr_imgs

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.
    
    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.
    
    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.
    
    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def video_group(vid_path):
    """
    Return the UCF50 group number encoded in a video path such as .../v_Biking_g07_c03.

    Clips that share a group share the same actor, background and viewpoint.

    Args:
        vid_path (str): Path to a video file or to its directory of frames.

    Returns:
        int: The group number.
    """
    return int(re.search(r'_g(\d+)_', os.path.basename(vid_path)).group(1))


def dataset_split(vid_dataset, val_groups, test_groups):
    """
    Split the dataset into training, validation, and test sets by UCF50 group.

    Clips in the same group show the same person in the same scene, so a split by clip puts
    near-duplicates on both sides and leaks them into evaluation. Here every group sits in
    exactly one split: validation and test take the groups listed, training takes the rest.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        val_groups (iterable of int): Group numbers held out for validation.
        test_groups (iterable of int): Group numbers held out for testing.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset), each a list of (video_path, label)
        tuples sorted by path.

    Raises:
        ValueError: If a group is assigned to both validation and test.
    """
    val_groups, test_groups = set(val_groups), set(test_groups)
    if val_groups & test_groups:
        raise ValueError(f'groups {sorted(val_groups & test_groups)} are in both val and test')
    print('Splitting train/validation/test datasets by group....')
    tr_dataset, val_dataset, ts_dataset = [], [], []
    for vid_path, label in sorted(vid_dataset.items()):
        group = video_group(vid_path)
        if group in test_groups:
            ts_dataset.append((vid_path, label))
        elif group in val_groups:
            val_dataset.append((vid_path, label))
        else:
            tr_dataset.append((vid_path, label))
    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    
    return padded_imgs, labels_tensor
