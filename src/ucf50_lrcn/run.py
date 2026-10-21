"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.
    
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import wandb

from ucf50_lrcn.video_datasets import VideoDataset, load_dataset, dataset_split
from ucf50_lrcn.utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from ucf50_lrcn.models import LRCN
from ucf50_lrcn.train import train
from ucf50_lrcn.evaluate import test, get_test_report, get_confusion_matrix

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
        -fd/--frame_dir: Directory for storing video frames.
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
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-vg', '--val_groups', type=int, nargs='+', default=[19, 20, 21],
                        help='UCF50 groups held out for validation')
    parser.add_argument('-tg', '--test_groups', type=int, nargs='+', default=[22, 23, 24, 25],
                        help='UCF50 groups held out for testing')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', choices=['lrcn'], default='lrcn',
                        help='Model architecture (only the LRCN is implemented)')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', type=str2bool, default=True,
                        help='Use the ImageNet-pretrained 2D CNN backbone (true/false)')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-nw', '--num_workers', type=int, default=0,
                        help='Worker processes that load clips in parallel with training')
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
    print('The overall test accuracy is {:.4f}% (test loss {:.4f}).'.format(100 * accuracy, loss))
    if wandb.run is not None:
        wandb.log({'test/loss': loss, 'test/acc': accuracy})
        wandb.run.summary['test/acc'] = accuracy
        wandb.run.summary['test/loss'] = loss
    return targets, outputs, accuracy, loss


def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    val_groups = args.val_groups
    test_groups = args.test_groups
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    num_workers = args.num_workers
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.define_metric('epoch')
        for key in ('lr', 'train/loss', 'train/acc', 'val/loss', 'val/acc'):
            wandb.define_metric(key, step_metric='epoch')

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                 n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone)

    if mode == 'train':
        # Load dataset and split into train/validation/test
        vid_dataset, label_dict = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(vid_dataset, val_groups, test_groups)

        # Save the splits for reproducibility and later use in evaluation
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        np.save('./splits.npy', splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type,
                                         num_workers)

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func, opt, lr_scheduler = build_optimization(model, learning_rate)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'

        # Main training procedure
        model.to(device)
        model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs)

        # Evaluate the best weights (train() returns them) on the held-out test groups
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        test_dl = test_dloaders(ts_dataset, batch_size, model_type, num_workers)['test']
        evaluate_split(model, test_dl, device)

    elif mode == 'eval':
        # Load saved dataset splits
        splits = np.load('./splits.npy', allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Load the trained model checkpoint and evaluate it on the test split
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        test_dl = test_dloaders(ts_dataset, batch_size, model_type, num_workers)['test']
        model.load_state_dict(torch.load(args.ckpt))
        model.to(device)
        targets, outputs, accuracy, _ = evaluate_split(model, test_dl, device)
        # Optionally, generate a detailed test report or confusion matrix:
        # print(get_test_report(targets, outputs, all_cats))
        # print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))

    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    args = args_parser()
    main(args)
