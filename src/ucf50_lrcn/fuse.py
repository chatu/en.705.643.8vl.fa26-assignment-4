"""
Module: fuse.py

Late fusion of the two streams: an LRCN trained on RGB frames and one trained on stacked
optical flow (python -m ucf50_lrcn.run --modality flow). Each model scores every clip of the
same group-based split, the two softmax outputs are averaged, and the accuracy of each stream
and of the fusion is reported on the validation and test groups.

Usage:
    python -m ucf50_lrcn.fuse --rgb_dir data/UCF50_frames --rgb_ckpt rgb.pt \
        --flow_dir data/UCF50_flow --flow_ckpt flow.pt --n_classes 50 --batch_size 8
"""

import argparse
import os

import torch
import wandb
from tqdm import tqdm

from ucf50_lrcn.flow import FLOW_MEAN, FLOW_STACK, FLOW_STD
from ucf50_lrcn.models import LRCN
from ucf50_lrcn.utils import (transform_stats, compose_data_transforms, compose_flow_transforms,
                              test_dloaders)
from ucf50_lrcn.video_datasets import VideoDataset, dataset_split, load_dataset


def predict_proba(model, dataloader, device):
    """
    Return the softmax output of a model for every clip in a dataloader, in order.

    Args:
        model (torch.nn.Module): Trained model, already on device.
        dataloader (DataLoader): Loader over the clips, not shuffled.
        device (torch.device): Evaluation device.

    Returns:
        tuple: (probs, targets), a (n_clips, n_classes) float tensor and a (n_clips,) tensor.
    """
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            probs.append(model(x_batch.to(device)).softmax(dim=1).cpu())
            targets.append(y_batch)
    return torch.cat(probs), torch.cat(targets)


def stream_splits(frame_dir, val_groups, test_groups):
    """
    Build the validation and test splits of one stream, keyed by video name.

    Args:
        frame_dir (str): Root of the stream's frame or flow folders.
        val_groups (list of int): Validation groups.
        test_groups (list of int): Test groups.

    Returns:
        dict: {'val': [...], 'test': [...]}, lists of (video_path, label) sorted by video name.
    """
    vid_dataset, _ = load_dataset(frame_dir)
    _, val_split, ts_split = dataset_split(vid_dataset, val_groups, test_groups)
    by_name = lambda split: sorted(split, key=lambda sample: os.path.basename(sample[0]))
    return {'val': by_name(val_split), 'test': by_name(ts_split)}


def load_stream(args, stream):
    """
    Build one stream's model from its checkpoint, with its eval transforms and channel count.

    Args:
        args (argparse.Namespace): Parsed arguments.
        stream (str): 'rgb' or 'flow'.

    Returns:
        tuple: (model, transforms, flow_channels)
    """
    h, w, mean, std = transform_stats('lrcn')
    if stream == 'flow':
        flow_channels, (_, transforms) = 2 * FLOW_STACK, compose_flow_transforms(h, w, FLOW_MEAN,
                                                                                 FLOW_STD)
    else:
        flow_channels, (_, transforms) = 0, compose_data_transforms(h, w, mean, std)
    model = LRCN(hidden_size=args.rnn_hidden_size, n_layers=args.rnn_n_layers, dropout_rate=0.0,
                 n_classes=args.n_classes, pretrained=False,
                 cnn_model=getattr(args, f'{stream}_backbone'),
                 temporal_pool=getattr(args, f'{stream}_pool'), in_channels=flow_channels or 3)
    model.load_state_dict(torch.load(getattr(args, f'{stream}_ckpt'), map_location='cpu'))
    return model, transforms, flow_channels


def args_parser(argv=None):
    """Parse the command-line arguments of the fusion evaluation."""
    parser = argparse.ArgumentParser(description='Late fusion of the RGB and flow LRCN streams')
    for stream in ('rgb', 'flow'):
        parser.add_argument(f'--{stream}_dir', required=True, help=f'Root of the {stream} folders')
        parser.add_argument(f'--{stream}_ckpt', required=True, help=f'{stream} model weights')
        parser.add_argument(f'--{stream}_backbone', default='resnet34', help=f'{stream} backbone')
        parser.add_argument(f'--{stream}_pool', choices=['last', 'mean'], default='last',
                            help=f'{stream} temporal pooling')
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--fr_per_vid', type=int, default=16)
    parser.add_argument('--rnn_hidden_size', type=int, default=100)
    parser.add_argument('--rnn_n_layers', type=int, default=1)
    parser.add_argument('--val_groups', type=int, nargs='+', default=[19, 20, 21])
    parser.add_argument('--test_groups', type=int, nargs='+', default=[22, 23, 24, 25])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--wandb_project', default=None,
                        help='Weights & Biases project to log to; logging is off if unset')
    parser.add_argument('--wandb_run_name', default=None)
    return parser.parse_args(argv)


def main(args):
    """Score both streams on the validation and test groups and report the fused accuracy."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    splits = {stream: stream_splits(getattr(args, f'{stream}_dir'), args.val_groups,
                                    args.test_groups) for stream in ('rgb', 'flow')}
    for split in ('val', 'test'):
        names = [[(os.path.basename(path), label) for path, label in splits[stream][split]]
                 for stream in ('rgb', 'flow')]
        if names[0] != names[1]:
            raise ValueError(f'the rgb and flow {split} splits hold different videos')

    probs, results = {}, {}
    for stream in ('rgb', 'flow'):
        model, transforms, flow_channels = load_stream(args, stream)
        model.to(device)
        for split in ('val', 'test'):
            dataset = VideoDataset(splits[stream][split], args.fr_per_vid, transforms,
                                   flow_channels)
            loader = test_dloaders(dataset, args.batch_size, 'lrcn', args.num_workers)['test']
            probs[stream, split], targets = predict_proba(model, loader, device)
            probs['targets', split] = targets
    for split in ('val', 'test'):
        targets = probs['targets', split]
        fused = (probs['rgb', split] + probs['flow', split]) / 2
        for name, scores in (('rgb', probs['rgb', split]), ('flow', probs['flow', split]),
                             ('fused', fused)):
            results[f'{split}/acc_{name}'] = (scores.argmax(dim=1) == targets).float().mean().item()
    for key, value in results.items():
        print(f'{key}: {100 * value:.2f}%')
    if wandb.run is not None:
        wandb.log(results)
        wandb.run.summary.update(results)
        wandb.finish()
    return results


if __name__ == '__main__':
    main(args_parser())
