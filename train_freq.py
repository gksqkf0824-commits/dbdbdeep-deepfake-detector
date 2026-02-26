"""
train_freq.py — Frequency Model Training (EfficientNet-V2-S, SRM×3 + Y-ch)

Key settings
------------
- Backbone    : EfficientNet-V2-S  (4-channel input)
- Input       : SRM residual ×3 + Y luminance ch — saved as .npy
- Augmentation: RandomHorizontalFlip + RandomVerticalFlip +
                RandomRotation(15°) + RandomErasing(p=0.4)
                + online Gaussian noise injection (p=0.3)
- Sampler     : WeightedRandomSampler  (class imbalance)
- Optimizer   : AdamW  lr=1e-4  weight_decay=0.15
- Scheduler   : OneCycleLR
- Loss        : BCEWithLogitsLoss
- Mixup       : alpha=0.4
- AMP         : autocast + GradScaler
- Batch size  : 48
- Epochs      : 30

Step 1 — Preprocess raw images to .npy  (run once)
----------------------------------------------------
  python train_freq.py preprocess --data /path/to/raw_dataset

Step 2 — Train
--------------
  python train_freq.py train --data /path/to/freq_data --save freq.pt

Directory structure (raw dataset)
----------------------------------
RAW_DATASET/
  train/ real/ *.jpg   fake/ *.jpg
  val/   real/ *.jpg   fake/ *.jpg
  test/  real/ *.jpg   fake/ *.jpg

After preprocessing, FREQ_DATA/ mirrors the same structure
but with *.npy files instead of *.jpg.
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm.auto import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1. SRM Preprocessing  (run once to convert images → .npy)
# ──────────────────────────────────────────────────────────────────────────────

def _get_srm_filters():
    f1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],
                   [0,-1,2,-1,0],[0,0,0,0,0]]) / 4.0
    f2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
                   [2,-6,8,-6,2],[-1,2,-2,2,-1]]) / 12.0
    f3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
                   [0,0,0,0,0],[0,0,0,0,0]]) / 2.0
    return [f1, f2, f3]


def process_4channel(img_path: str):
    """Raw image path → (224, 224, 4) uint8 numpy array."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y   = ycc[:, :, 0]

    srm_maps = []
    for f in _get_srm_filters():
        res = cv2.filter2D(y.astype(np.float32), -1, f)
        srm_maps.append(np.clip(res, 0, 255).astype(np.uint8))

    combined = np.concatenate([np.stack(srm_maps, axis=-1),
                                y[:, :, np.newaxis]], axis=-1)   # (H,W,4)
    return cv2.resize(combined, (224, 224), interpolation=cv2.INTER_CUBIC)


def preprocess_dataset(src_root: str, dst_root: str):
    """Convert all images in src_root tree to .npy files in dst_root."""
    from tqdm import tqdm as _tqdm
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            src_dir = os.path.join(src_root, split, cls)
            dst_dir = os.path.join(dst_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            if not os.path.exists(src_dir):
                print(f"[SKIP] {split}/{cls} not found")
                continue
            print(f"Processing {split}/{cls} ...")
            for fname in _tqdm(os.listdir(src_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                arr = process_4channel(os.path.join(src_dir, fname))
                if arr is not None:
                    save_name = os.path.splitext(fname)[0] + ".npy"
                    np.save(os.path.join(dst_dir, save_name), arr)
    print("Preprocessing complete.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FourChannelDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        self.split     = split
        self.transform = transform
        self.samples   = []
        self.class_counts = [0, 0]

        split_dir = os.path.join(root_dir, split)
        for cls_idx, cls_name in enumerate(["real", "fake"]):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
                    self.class_counts[cls_idx] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        data = np.load(fpath)

        # Online Gaussian noise (train only, p=0.3) — prevents frequency pattern memorisation
        if self.split == "train" and random.random() < 0.3:
            sigma = random.uniform(0.1, 1.0)
            noise = np.random.normal(0, sigma, data.shape).astype(np.float32)
            data  = np.clip(data.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        data   = data.astype(np.float32) / 255.0
        tensor = torch.from_numpy(data).permute(2, 0, 1)   # (4, 224, 224)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────────────────────────────────────

class FreqModel(nn.Module):
    def __init__(self):
        super().__init__()
        base     = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        with torch.no_grad():
            base.features[0][0].weight[:, :3] = old_conv.weight
            base.features[0][0].weight[:, 3]  = old_conv.weight.mean(dim=1) * 1.2

        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(in_features, 1),
        )
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Mixup
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


# ──────────────────────────────────────────────────────────────────────────────
# 5. Train / Evaluate
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        with autocast("cuda"):
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = FourChannelDataset(args.data, "train", TRAIN_TRANSFORM)
    val_ds   = FourChannelDataset(args.data, "val")
    test_ds  = FourChannelDataset(args.data, "test")

    # WeightedRandomSampler for class balance
    sample_weights = [
        1.0 / train_ds.class_counts[label]
        for _, label in train_ds.samples
    ]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model     = FreqModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.15)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
    )
    scaler = GradScaler("cuda")

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs   = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            imgs, la, lb, lam = mixup_data(imgs, labels, alpha=0.4)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(imgs)
                loss    = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_loss          = running_loss / len(train_loader)
        val_acc,  val_f1  = evaluate(model, val_loader,  device)
        test_acc, test_f1 = evaluate(model, test_loader, device)

        print(f"[Epoch {epoch+1:2d}/{args.epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"Val  Acc/F1: {val_acc:.4f}/{val_f1:.4f} | "
              f"Test Acc/F1: {test_acc:.4f}/{test_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.save)
            print(f"  → Saved best model  (Val F1: {best_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")
    print(f"Weights saved to: {args.save}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Frequency model training")
    sub    = parser.add_subparsers(dest="cmd")

    # --- preprocess sub-command ---
    pre = sub.add_parser("preprocess", help="Convert raw images to .npy SRM arrays")
    pre.add_argument("--data", required=True, help="Raw dataset root")
    pre.add_argument("--out",  default="freq_data", help="Output directory for .npy files")

    # --- train sub-command ---
    tr = sub.add_parser("train", help="Train the frequency model")
    tr.add_argument("--data",   default="freq_data", help="Preprocessed .npy dataset root")
    tr.add_argument("--save",   default="freq.pt",   help="Output weight file")
    tr.add_argument("--epochs", type=int,   default=30)
    tr.add_argument("--batch",  type=int,   default=48)
    tr.add_argument("--lr",     type=float, default=1e-4)

    args = parser.parse_args()

    if args.cmd == "preprocess":
        preprocess_dataset(args.data, args.out)
    elif args.cmd == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
