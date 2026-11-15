from collections import Counter
import torch
from torch import nn
from torchvision import models
from loaders import get_loaders
from train_utils import run_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, test_loader, classes = get_loaders()

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, len(classes))
model.load_state_dict(torch.load("pets_resnet18.pt", map_location=device))
model = model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()
val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)

test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device)
print(test_loss, test_acc)
