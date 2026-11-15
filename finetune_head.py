import torch
from torch import nn
from torchvision import models
from loaders import get_loaders
from train_utils import run_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, test_loader, classes = get_loaders()

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False          # 1. freeze everything
model.fc = nn.Linear(512, len(classes))  # 2. new head (switch on)
for param in model.layer4.parameters():
    param.requires_grad = True           # 3. unfreeze the last block
model = model.to(device)

optimizer = torch.optim.Adam([           # 4. two step sizes
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

for epoch in range(5):
    train_loss, train_acc = run_epoch(
        model, train_loader, loss_fn, device, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
    print(f"epoch {epoch + 1}: train loss {train_loss:.3f} acc {train_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")
