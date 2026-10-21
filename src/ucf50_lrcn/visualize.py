"""
Module: visualize.py

Save image grids of clips, to check by eye what the model receives: one row per clip, its
sampled frames left to right in temporal order, titled with the true label and, after
training, the predicted label.

Usage:
    python -m ucf50_lrcn.visualize --frame_dir data/UCF50_frames --split train --out clips.png
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch

from ucf50_lrcn.models import LRCN
from ucf50_lrcn.utils import compose_data_transforms, transform_stats
from ucf50_lrcn.video_datasets import VideoDataset, dataset_split, load_dataset

plt.switch_backend('Agg')  # render to files; no display is needed or assumed


def save_clip_grid(samples, class_names, fr_per_vid, out_path, predictions=None):
    """
    Save a grid with one row per clip and its frames in temporal order.

    Args:
        samples (list of tuple): (video_path, label) pairs to draw.
        class_names (list of str): Class name for each label index.
        fr_per_vid (int): Frames sampled per clip, as the model would see them.
        out_path (str): Where to write the PNG.
        predictions (list of int, optional): Predicted label per sample, shown in each title.
    """
    dataset = VideoDataset(samples, fr_per_vid)
    fig, axes = plt.subplots(len(samples), fr_per_vid,
                             figsize=(1.6 * fr_per_vid, 1.8 * len(samples)), squeeze=False)
    for row, (clip, label) in enumerate(dataset[i] for i in range(len(samples))):
        title = f'{class_names[label]} | {os.path.basename(samples[row][0])}'
        if predictions is not None:
            mark = 'correct' if predictions[row] == label else 'WRONG'
            title += f' | predicted {class_names[predictions[row]]} ({mark})'
        for col in range(fr_per_vid):
            axes[row, col].axis('off')
            if col < len(clip):
                axes[row, col].imshow(clip[col].permute(1, 2, 0).numpy())
                axes[row, col].set_title(f't{col}', fontsize=6)
        axes[row, 0].text(0, 1.35, title, transform=axes[row, 0].transAxes, fontsize=8)
    # Fixed spacing rather than tight_layout: the row titles are wider than a short row, and
    # tight_layout shrank every frame to a speck to make room for them.
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.05, hspace=0.9)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def sample_per_class(split, class_names, n_classes, seed=0):
    """
    Pick one random clip from each of n_classes random classes in a split.

    Args:
        split (list of tuple): (video_path, label) pairs.
        class_names (list of str): Class name for each label index.
        n_classes (int): How many classes to draw.
        seed (int): Random seed, so the same clips are drawn each time.

    Returns:
        list of tuple: (video_path, label) pairs, one per chosen class.
    """
    rng = random.Random(seed)
    chosen = rng.sample(range(len(class_names)), min(n_classes, len(class_names)))
    return [rng.choice([s for s in split if s[1] == label]) for label in sorted(chosen)]


def predict(samples, ckpt, n_classes, fr_per_vid, cnn_backbone):
    """
    Predict the label of each clip with a trained RGB LRCN.

    Args:
        samples (list of tuple): (video_path, label) pairs.
        ckpt (str): Path to the model weights saved by training.
        n_classes (int): Number of classes the model was trained on.
        fr_per_vid (int): Frames sampled per clip.
        cnn_backbone (str): Backbone the model was trained with.

    Returns:
        list of int: Predicted label per sample.
    """
    h, w, mean, std = transform_stats('lrcn')
    _, transforms = compose_data_transforms(h, w, mean, std)
    model = LRCN(hidden_size=100, n_layers=1, dropout_rate=0.0, n_classes=n_classes,
                 pretrained=False, cnn_model=cnn_backbone)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    dataset = VideoDataset(samples, fr_per_vid, transforms)
    with torch.no_grad():
        return [int(model(dataset[i][0].unsqueeze(0)).argmax()) for i in range(len(samples))]


def main():
    """Draw one clip per class from the chosen split of the group-based UCF50 split."""
    parser = argparse.ArgumentParser(description='Save a grid of clips as the model sees them')
    parser.add_argument('--frame_dir', required=True, help='Root of the extracted frame folders')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--n_classes', type=int, default=10, help='Classes to draw, one clip each')
    parser.add_argument('--fr_per_vid', type=int, default=16)
    parser.add_argument('--out', default='clips.png')
    parser.add_argument('--ckpt', default=None,
                        help='Trained RGB model weights; if given, titles show its predictions')
    parser.add_argument('--cnn_backbone', default='resnet34', help='Backbone of --ckpt')
    args = parser.parse_args()

    vid_dataset, label_dict = load_dataset(args.frame_dir)
    class_names = sorted(label_dict, key=label_dict.get)
    splits = dict(zip(['train', 'val', 'test'],
                      dataset_split(vid_dataset, [19, 20, 21], [22, 23, 24, 25])))
    samples = sample_per_class(splits[args.split], class_names, args.n_classes)
    predictions = (predict(samples, args.ckpt, len(class_names), args.fr_per_vid,
                           args.cnn_backbone) if args.ckpt else None)
    save_clip_grid(samples, class_names, args.fr_per_vid, args.out, predictions)
    print(f'saved {len(samples)} clips from the {args.split} split to {args.out}')


if __name__ == '__main__':
    main()
