"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model
parameters, and training hyperparameters. Depending on the selected mode ('train' or 'eval'),
it performs the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the LRCN and defines the loss function, optimizer, and learning rate
      scheduler.
    - Runs the training loop and saves the best model weights (by validation accuracy) to
      <out_dir>/models, then evaluates them on the test split.

- Eval mode:
    - Loads the dataset splits saved in <out_dir> by the training run.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.

The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import wandb

from ucf50_lrcn.video_datasets import VideoDataset, load_dataset, dataset_split
from ucf50_lrcn.utils import (transform_stats, compose_data_transforms, compose_flow_transforms,
                              train_val_dloaders, test_dloaders)
from ucf50_lrcn.flow import FLOW_MEAN, FLOW_STACK, FLOW_STD
from ucf50_lrcn.models import LRCN
from ucf50_lrcn.train import train
from ucf50_lrcn.evaluate import test

def str2bool(value):
    """
    Parse a command-line truth value.

    Args:
        value (str): One of true/false, yes/no, 1/0 (case-insensitive).

    Returns:
        bool: The parsed value.

    Raises:
        argparse.ArgumentTypeError: If the value is not a recognised truth value.
    """
    lowered = str(value).strip().lower()
    if lowered in ('true', 'yes', '1'):
        return True
    if lowered in ('false', 'no', '0'):
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, got {value!r}')


def args_parser(argv=None):
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Args:
        argv (list of str, optional): Arguments to parse; defaults to sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments include:
        -fd/--frame_dir: Root of the frame (or, with --modality flow, flow) folders to read.
        -vg/--val_groups: UCF50 groups held out for validation (default 19 20 21).
        -tg/--test_groups: UCF50 groups held out for testing (default 22 23 24 25).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type; only 'lrcn' is implemented (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
        --modality: 'rgb' frames or 'flow' stacks (default 'rgb').
        -tp/--temporal_pool: 'last' LSTM step or 'mean' over steps (default 'last').
        -nw/--num_workers: DataLoader worker processes (default 0).
        --out_dir: Where models/ and splits.npy are written and read (default '.').
        --wandb_project, --wandb_run_name: Weights & Biases logging, off unless a project is set.
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Root of the frame or flow folders to read')
    parser.add_argument('-vg', '--val_groups', type=int, nargs='+', default=[19, 20, 21],
                        help='UCF50 groups held out for validation')
    parser.add_argument('-tg', '--test_groups', type=int, nargs='+', default=[22, 23, 24, 25],
                        help='UCF50 groups held out for testing')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, required=True,
                        help='Number of classes for the classification task')

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', choices=['lrcn'], default='lrcn',
                        help='Model architecture (only the LRCN is implemented)')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                        help='2D CNN backbone: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', type=str2bool, default=True,
                        help='Use the ImageNet-pretrained 2D CNN backbone (true/false)')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100,
                        help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('--modality', choices=['rgb', 'flow'], default='rgb',
                        help='rgb frames, or flow stacks from ucf50_lrcn.flow '
                             '(frame_dir must match)')
    parser.add_argument('-tp', '--temporal_pool', choices=['last', 'mean'], default='last',
                        help="Classify from the last LSTM step, or average every step's logits")
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1,
                        help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, required=True,
                        help="Either 'train' or 'eval'")
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5,
                        help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-nw', '--num_workers', type=int, default=0,
                        help='Worker processes that load clips in parallel with training')
    parser.add_argument('--out_dir', default='.',
                        help='Where train mode writes models/ and splits.npy and eval mode reads '
                             'splits.npy')
    parser.add_argument('--wandb_project', default=None,
                        help='Weights & Biases project to log to; logging is off if unset')
    parser.add_argument('--wandb_run_name', default=None, help='Weights & Biases run name')

    args = parser.parse_args(argv)
    return args

def build_optimization(model, learning_rate):
    """
    Create the loss function, optimizer and learning rate scheduler for training.

    The loss sums over the batch; train.get_epoch_loss divides the epoch total by the number
    of videos, so the reported loss is the mean per video.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        learning_rate (float): Initial learning rate.

    Returns:
        tuple: (loss_func, optimizer, scheduler)
    """
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    # ReduceLROnPlateau no longer accepts verbose= (removed from PyTorch); passing it raised a
    # TypeError before the first epoch. train() already prints the learning rate each epoch.
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    return loss_func, opt, lr_scheduler


def evaluate_split(model, dataloader, device):
    """
    Evaluate a model on a split, print the result and log it to W&B if a run is active.

    Args:
        model (torch.nn.Module): Trained model, already on device.
        dataloader (DataLoader): Loader over the split, here the test split.
        device (torch.device): Evaluation device.

    Returns:
        tuple: (targets, outputs, accuracy, loss)
    """
    targets, outputs, accuracy, loss = test(model, dataloader, device,
                                            nn.CrossEntropyLoss(reduction='sum'))
    print(f'The overall test accuracy is {100 * accuracy:.4f}% (test loss {loss:.4f}).')
    if wandb.run is not None:
        wandb.log({'test/loss': loss, 'test/acc': accuracy})
        wandb.run.summary['test/acc'] = accuracy
        wandb.run.summary['test/loss'] = loss
    return targets, outputs, accuracy, loss


def build_transforms(args):
    """
    Return the transforms and per-frame channel count for the chosen modality.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: (flow_channels, train_transforms, val_test_transforms); flow_channels is 0 for RGB.
    """
    h, w, mean, std = transform_stats(args.model_type)
    if args.modality == 'flow':
        return (2 * FLOW_STACK,) + compose_flow_transforms(h, w, FLOW_MEAN, FLOW_STD)
    return (0,) + compose_data_transforms(h, w, mean, std)


def run_train(args, model, transforms, device):
    """
    Split by group, train with validation, save the splits, and test the best weights.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): Freshly built model.
        transforms (tuple): (flow_channels, train_transforms, val_test_transforms).
        device (torch.device): Training device.
    """
    flow_channels, tr_transforms, val_ts_transforms = transforms
    vid_dataset, _ = load_dataset(args.frame_dir)
    tr_split, val_split, ts_split = dataset_split(vid_dataset, args.val_groups, args.test_groups)

    # Save the splits for reproducibility and later use in evaluation
    model_dir = os.path.join(args.out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'splits.npy'),
            {'train': np.array(tr_split), 'val': np.array(val_split), 'test': np.array(ts_split)})

    dataloaders = train_val_dloaders(
        VideoDataset(tr_split, args.fr_per_vid, tr_transforms, flow_channels),
        VideoDataset(val_split, args.fr_per_vid, val_ts_transforms, flow_channels),
        args.batch_size, args.model_type, args.num_workers)
    loss_func, opt, lr_scheduler = build_optimization(model, args.learning_rate)
    model.to(device)
    model, _, _ = train(dataloaders, model, loss_func, opt, lr_scheduler, device, model_dir,
                        args.n_epochs)

    # Evaluate the best weights (train() returns them) on the held-out test groups
    ts_dataset = VideoDataset(ts_split, args.fr_per_vid, val_ts_transforms, flow_channels)
    evaluate_split(model, test_dloaders(ts_dataset, args.batch_size, args.model_type,
                                        args.num_workers)['test'], device)


def run_eval(args, model, transforms, device):
    """
    Evaluate a saved checkpoint on the test split saved by the training run.

    Args:
        args (argparse.Namespace): Parsed command-line arguments; args.ckpt is the checkpoint.
        model (torch.nn.Module): Model with the checkpoint's architecture.
        transforms (tuple): (flow_channels, train_transforms, val_test_transforms).
        device (torch.device): Evaluation device.
    """
    flow_channels, _, val_ts_transforms = transforms
    splits = np.load(os.path.join(args.out_dir, 'splits.npy'), allow_pickle=True)
    ts_split = [(sample[0], int(sample[1])) for sample in splits.item()['test']]
    ts_dataset = VideoDataset(ts_split, args.fr_per_vid, val_ts_transforms, flow_channels)
    # map_location: checkpoints saved during GPU training must also load on a CPU-only machine
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.to(device)
    evaluate_split(model, test_dloaders(ts_dataset, args.batch_size, args.model_type,
                                        args.num_workers)['test'], device)


def main(args):
    """
    Train or evaluate an LRCN as configured by the command-line arguments.

    Train mode splits the data by group, trains with validation, saves the best weights to
    <out_dir>/models and the splits to <out_dir>/splits.npy, then reports test accuracy of the
    best weights.
    Eval mode loads a checkpoint and reports its accuracy on the saved test split.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ValueError: If args.mode is neither 'train' nor 'eval'.
    """
    if args.mode not in ('train', 'eval'):
        raise ValueError('The mode argument must be either "train" or "eval".')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.define_metric('epoch')
        for key in ('lr', 'train/loss', 'train/acc', 'val/loss', 'val/acc'):
            wandb.define_metric(key, step_metric='epoch')

    transforms = build_transforms(args)
    model = LRCN(hidden_size=args.rnn_hidden_size, n_layers=args.rnn_n_layers,
                 dropout_rate=args.dropout, n_classes=args.n_classes,
                 # ImageNet weights only matter for training; in eval the checkpoint
                 # replaces them, so skip a download that fails without network access.
                 pretrained=args.pretrained and args.mode == 'train', cnn_model=args.cnn_backbone,
                 temporal_pool=args.temporal_pool, in_channels=transforms[0] or 3)
    if args.mode == 'train':
        run_train(args, model, transforms, device)
    else:
        run_eval(args, model, transforms, device)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main(args_parser())
