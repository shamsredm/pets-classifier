import torch
from torch import nn
from loaders import get_loaders
from train_utils import run_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, test_loader, classes = get_loaders()


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))


model = nn.Sequential(
    block(3, 16), block(16, 32), block(32, 64),   # 224 -> 112 -> 56 -> 28
    nn.Flatten(),
    nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
    nn.Linear(128, len(classes)),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    train_loss, train_acc = run_epoch(
        model, train_loader, loss_fn, device, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
    print(f"epoch {epoch + 1}: train loss {train_loss:.3f} acc {train_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")
