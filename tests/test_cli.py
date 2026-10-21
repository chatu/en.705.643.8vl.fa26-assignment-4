"""Regression tests for the command-line interface."""

import pytest
import torch

from ucf50_lrcn.run import args_parser, build_optimization

REQUIRED = ["--n_classes", "50", "--batch_size", "4", "--mode", "train"]


@pytest.mark.parametrize("value, expected", [("False", False), ("True", True)])
def test_pretrained_flag_is_parsed_as_a_boolean(value, expected):
    """Regression: --pretrained had no type, so the string 'False' was truthy and a
    pretrained backbone was loaded whatever the user asked for."""
    args = args_parser(REQUIRED + ["--pretrained", value])
    assert args.pretrained is expected


def test_training_setup_builds_on_current_pytorch():
    """Regression: ReduceLROnPlateau(verbose=1) raised TypeError on current PyTorch, so
    training crashed before the first epoch."""
    model = torch.nn.Linear(4, 2)
    loss_func, optimizer, scheduler = build_optimization(model, 1e-3)
    loss = loss_func(model(torch.randn(3, 4)), torch.tensor([0, 1, 0]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    assert optimizer.param_groups[0]["lr"] == 1e-3
