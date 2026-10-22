"""Regression test for the training loop."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ucf50_lrcn.train import train


def test_a_checkpoint_is_saved_even_if_validation_accuracy_stays_zero(tmp_path):
    """Regression: the best validation accuracy started at 0.0 and only a strictly better
    epoch was saved, so a run that never got a validation clip right wrote no checkpoint
    and test.sh / fuse.sh then failed on a missing file."""
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([-10.0, 10.0]))   # always predicts class 1
    data = TensorDataset(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
    loaders = {'train': DataLoader(data, batch_size=2), 'val': DataLoader(data, batch_size=2)}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    train(loaders, model, torch.nn.CrossEntropyLoss(reduction='sum'), optimizer, scheduler,
          torch.device('cpu'), str(tmp_path), n_epochs=1)
    assert (tmp_path / 'best_model_wts.pt').exists()
