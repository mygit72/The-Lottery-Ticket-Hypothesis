import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.trainer import Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def make_toy_loader(n=32):
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0).long()  # linearly separable toy task
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_evaluate_reports_accuracy_in_zero_to_one():
    model = TinyModel()
    loader = make_toy_loader()
    trainer = Trainer(model, loader, loader, loader, device='cpu')

    loss, acc = trainer.evaluate(loader)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0


def test_train_improves_loss_on_separable_toy_task():
    model = TinyModel()
    loader = make_toy_loader()
    trainer = Trainer(model, loader, loader, loader, device='cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    pre_loss, _ = trainer.evaluate(loader)
    trainer.train(optimizer, epochs=20, verbose=False)
    post_loss, post_acc = trainer.evaluate(loader)

    assert post_loss < pre_loss
    assert post_acc > 0.9


def test_masks_freeze_pruned_weights_during_training():
    model = TinyModel()
    loader = make_toy_loader()
    trainer = Trainer(model, loader, loader, loader, device='cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    mask = torch.ones_like(model.fc.weight)
    mask[0, :] = 0  # freeze the first output neuron's weights
    with torch.no_grad():
        model.fc.weight.mul_(mask)
    unfrozen_row_before = model.fc.weight.data[1, :].clone()

    trainer.train(optimizer, epochs=5, masks={'fc': mask}, verbose=False)

    assert torch.equal(model.fc.weight.data[0, :], torch.zeros(4))
    # Unfrozen row should have moved away from its initial value
    assert not torch.equal(model.fc.weight.data[1, :], unfrozen_row_before)
