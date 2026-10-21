"""Regression tests for the command-line interface."""

import pytest

from ucf50_lrcn.run import args_parser

REQUIRED = ["--n_classes", "50", "--batch_size", "4", "--mode", "train"]


@pytest.mark.parametrize("value, expected", [("False", False), ("True", True)])
def test_pretrained_flag_is_parsed_as_a_boolean(value, expected):
    """Regression: --pretrained had no type, so the string 'False' was truthy and a
    pretrained backbone was loaded whatever the user asked for."""
    args = args_parser(REQUIRED + ["--pretrained", value])
    assert args.pretrained is expected
