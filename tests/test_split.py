"""Regression tests for the train/validation/test split (the data leak)."""

import re

import pytest

from ucf50_lrcn.video_datasets import dataset_split

CLASSES = ["Biking", "Diving", "Punch"]


def _group(path):
    """Group number encoded in a UCF50 file name such as v_Biking_g07_c03."""
    return int(re.search(r"_g(\d+)_", path).group(1))


@pytest.fixture(name="vid_dataset")
def fixture_vid_dataset():
    """A UCF50-shaped dataset: 3 classes x 25 groups x 4 clips, as {path: label}."""
    return {f"UCF50/{name}/v_{name}_g{g:02d}_c{c:02d}": label
            for label, name in enumerate(CLASSES)
            for g in range(1, 26) for c in range(1, 5)}


def test_no_group_appears_in_more_than_one_split(vid_dataset):
    """Regression: a per-clip random split put clips of the same person and scene
    (the same group) on both sides, which leaked them into validation and test."""
    splits = dataset_split(vid_dataset, val_groups=[19, 20, 21], test_groups=[22, 23, 24, 25])
    groups = [{_group(path) for path, _ in split} for split in splits]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_every_clip_is_assigned_exactly_once(vid_dataset):
    """The three splits partition the dataset."""
    splits = dataset_split(vid_dataset, val_groups=[19, 20, 21], test_groups=[22, 23, 24, 25])
    paths = [path for split in splits for path, _ in split]
    assert sorted(paths) == sorted(vid_dataset)
