"""Regression tests for the LRCN model."""

import pytest
import torch

from ucf50_lrcn.models import LRCN

N_CLASSES = 50


@pytest.fixture(name="model")
def fixture_model():
    """A small, randomly initialised LRCN in eval mode, so dropout cannot add noise."""
    torch.manual_seed(0)
    return LRCN(hidden_size=16, n_layers=1, dropout_rate=0.0, n_classes=N_CLASSES,
                pretrained=False, cnn_model="resnet18").eval()


def test_output_has_one_row_of_logits_per_video(model):
    """(batch, time, C, H, W) in, (batch, n_classes) out."""
    with torch.no_grad():
        out = model(torch.randn(3, 4, 3, 64, 64))
    assert out.shape == (3, N_CLASSES)


def test_prediction_does_not_depend_on_other_videos_in_the_batch(model):
    """Regression: the LSTM was built without batch_first=True and fed (batch, 1, features),
    so it ran along the batch dimension and mixed information between unrelated videos."""
    torch.manual_seed(1)
    videos = torch.randn(3, 4, 3, 64, 64)
    altered = videos.clone()
    altered[0] = torch.randn_like(altered[0])
    with torch.no_grad():
        before, after = model(videos), model(altered)
    torch.testing.assert_close(after[1:], before[1:])


def test_resnet152_backbone_is_a_resnet152():
    """Regression: asking for resnet152 silently built a resnet34."""
    model = LRCN(hidden_size=8, n_layers=1, dropout_rate=0.0, n_classes=N_CLASSES,
                 pretrained=False, cnn_model="resnet152")
    assert model.rnn.input_size == 2048
    n_blocks = sum(len(getattr(model.base_model, f"layer{i}")) for i in range(1, 5))
    assert n_blocks == 50  # a resnet34 has 16
