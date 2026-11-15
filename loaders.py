from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

root = Path(__file__).parent / "data"

# Pretrained models expect 224x224 images, normalised with these numbers
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_loaders(batch_size=32):
    full_train = datasets.OxfordIIITPet(
        root=root, split="trainval", transform=transform, download=False
    )
    test_data = datasets.OxfordIIITPet(
        root=root, split="test", transform=transform, download=False
    )

    # hold back 20% of trainval for validation
    n_val = int(0.2 * len(full_train))
    n_train = len(full_train) - n_val
    train_data, val_data = random_split(
        full_train, [
            n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader, full_train.classes
