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
