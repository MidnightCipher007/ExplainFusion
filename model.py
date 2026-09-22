import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from utils_preprocess import preprocess_medical

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# TRANSFORM
# Applied AFTER preprocess_medical (which handles resize + CLAHE)
# Must match train.py exactly.
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------------------
# FIX DENSENET FOR GRADCAM (avoids inplace ReLU issues)
# ---------------------------
def patch_densenet_forward(model):
    def new_forward(x):
        features = model.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        return out

    model.forward = new_forward


def disable_inplace(model):
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


# ---------------------------
# LOAD MODEL
# ---------------------------
def load_model(path="model.pth"):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    model.load_state_dict(torch.load(path, map_location=device))

    disable_inplace(model)
    patch_densenet_forward(model)

    model = model.to(device)
    model.eval()

    return model


# ---------------------------
# PREDICTION
# ---------------------------
def predict(model, image):
    """
    image: raw PIL image from Gradio (any size, any mode)
    Returns: label, confidence, input_tensor, probs_np, processed_pil
    """
    # Step 1: medical preprocessing (CLAHE, resize to 224, grayscale→3ch)
    processed = preprocess_medical(image)

    # Step 2: torchvision transform (ToTensor + ImageNet normalize)
    tensor = transform(processed).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    probs_np = probs.cpu().numpy()[0]

    p_normal = float(probs_np[0])
    p_pneumonia = float(probs_np[1])

    label_idx = int(probs_np.argmax())
    label = "Pneumonia" if label_idx == 1 else "Normal"
    confidence = float(probs_np[label_idx])

    # Soft uncertainty penalty when model is unsure
    diff = abs(p_pneumonia - p_normal)
    if diff < 0.05:
        confidence *= 0.80
    elif diff < 0.10:
        confidence *= 0.90

    confidence = max(0.0, min(confidence, 1.0))

    return label, confidence, tensor, probs_np, processed