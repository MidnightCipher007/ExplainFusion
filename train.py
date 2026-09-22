import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
from utils_preprocess import preprocess_medical

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------
# CUSTOM DATASET WITH MEDICAL PREPROCESSING
# ---------------------------
# We wrap ImageFolder with preprocess_medical BEFORE the torchvision transform
# This ensures train and inference see the same pixel distribution.
class MedicalImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        img = preprocess_medical(img)          # CLAHE + normalize → PIL
        if self.transform:
            img = self.transform(img)          # ToTensor + ImageNet norm
        return img, label


# ---------------------------
# TRANSFORMS (after preprocess_medical, image is already 224x224)
# ---------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------------------
# DATASET
# Note: Kaggle chest X-ray val folder has only 16 images — too small.
# Option A: use the provided val folder anyway (quick)
# Option B: split train 80/20 for a proper val set (recommended ✅)
# ---------------------------
USE_SPLIT_VAL = True   # Set False to use dataset/val folder directly

if USE_SPLIT_VAL:
    full_train = MedicalImageFolder("dataset/train", transform=train_transform)
    val_size = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size

    # We need separate transforms for train vs val after splitting
    # Simplest approach: use train transform for both, then override
    train_data, val_data_raw = random_split(full_train, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))

    # val should not have augmentation — we wrap with val_transform manually
    class ValSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            img_tensor, label = self.subset[idx]
            # img_tensor is already transformed; we need raw image
            # Better: re-load from path
            path, lbl = self.subset.dataset.samples[self.subset.indices[idx]]
            img = Image.open(path).convert("RGB")
            img = preprocess_medical(img)
            img = self.transform(img)
            return img, lbl

        def __len__(self):
            return len(self.subset)

    val_data = ValSubset(train_data, val_transform)

    # Rebuild train_data with proper transform (already has train_transform)
    targets = [full_train.targets[i] for i in train_data.indices]

else:
    train_data = MedicalImageFolder("dataset/train", transform=train_transform)
    val_data = MedicalImageFolder("dataset/val", transform=val_transform)
    targets = train_data.targets

print("Classes:", train_data.dataset.classes if USE_SPLIT_VAL else train_data.classes)

# ---------------------------
# WEIGHTED SAMPLER (handles class imbalance)
# Kaggle dataset is ~3:1 pneumonia:normal — critical to fix
# ---------------------------
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

print(f"Class counts: {class_counts}")
print(f"Sampler weights: Normal={class_weights[0]:.4f}, Pneumonia={class_weights[1]:.4f}")

# ---------------------------
# DATALOADER
# ---------------------------
train_loader = DataLoader(
    train_data,
    batch_size=16,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ---------------------------
# LOSS WEIGHTS (extra safety for imbalance)
# ---------------------------
total = sum(class_counts)
loss_weights = torch.tensor([
    total / class_counts[0],
    total / class_counts[1]
]).float().to(device)
print("Loss weights:", loss_weights)

# ---------------------------
# MODEL
# ---------------------------
model = densenet121(weights=DenseNet121_Weights.DEFAULT)

model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

model = model.to(device)

# ---------------------------
# LOSS + OPTIMIZER + SCHEDULER
# ---------------------------
criterion = nn.CrossEntropyLoss(weight=loss_weights)

optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ---------------------------
# AMP
# ---------------------------
use_cuda = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda" if use_cuda else "cpu")


# ---------------------------
# TRAIN LOOP
# ---------------------------
def train_model(epochs=15):
    best_val_acc = 0
    patience = 5
    trigger = 0

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        # TRAIN
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda" if use_cuda else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total

        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        normal_acc = (all_preds[all_labels==0] == 0).mean()
        pneumonia_acc = (all_preds[all_labels==1] == 1).mean()

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Normal Acc: {normal_acc:.4f} | Pneumonia Acc: {pneumonia_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            print("✅ Saved Best Model")
            trigger = 0
        else:
            trigger += 1

        if trigger >= patience:
            print("⛔ Early stopping triggered")
            break

    print(f"\n🎉 Training Complete | Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model()