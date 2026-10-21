"""Test that preprocessing produces what the dataset loader expects."""

import cv2
import numpy as np

from ucf50_lrcn.preprocess import preprocess
from ucf50_lrcn.video_datasets import VideoDataset, load_dataset


def _write_numbered_video(path, n_frames=40):
    """An .avi whose frame i is a flat grey image of brightness 5 * i."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (32, 32))
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), 5 * i, dtype=np.uint8))
    writer.release()


def test_preprocessed_frames_load_with_folder_labels_and_full_clips(tmp_path):
    """Videos become frame folders that load_dataset and VideoDataset read correctly; labels
    follow the class folder even when the file name misspells the class (as UCF50 does)."""
    _write_numbered_video(tmp_path / "UCF50" / "Billiards" / "v_Billards_g01_c01.avi")
    _write_numbered_video(tmp_path / "UCF50" / "Biking" / "v_Biking_g01_c01.avi")

    short = preprocess(str(tmp_path / "UCF50"), str(tmp_path / "frames"), n_frames=32, workers=1)
    vid_dataset, label_dict = load_dataset(str(tmp_path / "frames"))

    assert short == []
    assert label_dict == {"Biking": 0, "Billiards": 1}
    billiards = [path for path in vid_dataset if "Billards" in path]
    assert vid_dataset[billiards[0]] == label_dict["Billiards"]

    clip, _ = VideoDataset([(billiards[0], 1)], fr_per_vid=8)[0]
    brightness = [int(round(float(frame.float().mean()) / 5.0)) for frame in clip]
    assert len(brightness) == 8
    assert brightness[0] == 0 and brightness[-1] == 39
