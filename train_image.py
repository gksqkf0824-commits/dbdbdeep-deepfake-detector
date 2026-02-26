"""
train_image.py — Image Model Training (EfficientNet-V2-S, RGB)
Experiment 2 configuration.

Key settings
------------
- Backbone    : EfficientNet-V2-S (ImageNet pretrained)
- Input       : RGB 3-ch, 224×224
- Augmentation: ColorJitter + RandomAffine + GaussianBlur + RandomErasing
- Optimizer   : AdamW  lr=8e-5  weight_decay=0.01
- Loss        : BCEWithLogitsLoss + Label Smoothing (α=0.1)
- Batch size  : 128
- Epochs      : 5

Directory structure expected
-----------------------------
DATASET_ROOT/
  train/
    real/  *.jpg
    fake/  [sub-dirs OK] *.jpg
  val/
    real/  ...
    fake/  ...
  test/
    real/  ...
    fake/  ...

Usage
-----
  python train_image.py --data /path/to/dataset --save image.pth
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        self.transform = transform
        self.samples   = []
        split_dir = os.path.join(root_dir, split)

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for dirpath, _, files in os.walk(cls_dir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(dirpath, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────────────────────────────────────

LABEL_SMOOTHING = 0.1


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device)
        labels = labels.to(device).view(-1, 1)
        # Label smoothing: 1 → 0.95, 0 → 0.05
        smooth_labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, smooth_labels)
        loss.backward()
        optimizer.step()

        preds    = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        running_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device).view(-1, 1)
        preds  = (torch.sigmoid(model(imgs)) > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset",   help="Path to dataset root")
    parser.add_argument("--save",   default="image.pth", help="Output weight file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--lr",     type=float, default=8e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = DeepfakeImageDataset(args.data, "train", TRAIN_TRANSFORM)
    val_ds   = DeepfakeImageDataset(args.data, "val",   EVAL_TRANSFORM)
    test_ds  = DeepfakeImageDataset(args.data, "test",  EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model     = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc,  val_f1  = evaluate(model, val_loader,  device)
        test_acc, test_f1 = evaluate(model, test_loader, device)

        print(f"[Epoch {epoch+1:2d}/{args.epochs}] "
              f"Loss: {loss:.4f} | TrainAcc: {train_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.save)
            print(f"  → Saved best model  (Val F1: {best_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    print(f"Weights saved to: {args.save}")


if __name__ == "__main__":
    main()
