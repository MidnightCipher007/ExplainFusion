# run_evaluation.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import load_model, device
from evaluate import evaluate

# same transform as validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_data = datasets.ImageFolder("dataset/val", transform=val_transform)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

model = load_model("model.pth")

evaluate(model, val_loader, device, class_names=val_data.classes)