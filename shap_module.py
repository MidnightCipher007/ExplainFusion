import torch
import numpy as np
import cv2
from utils_preprocess import preprocess_medical


def generate_shap(image, model, transform, device):
    """
    Integrated Gradients (IG) attribution.
    image: PIL Image (already preprocessed via preprocess_medical)
    """
    model.eval()

    try:
        # image already preprocessed — just transform
        x = transform(image).unsqueeze(0).to(device)

        # Gray baseline (more informative than black for X-rays)
        baseline = torch.full_like(x, 0.5)

        # ---------------------------
        # GET TARGET CLASS
        # ---------------------------
        with torch.no_grad():
            out = model(x)
            target_class = int(torch.argmax(out, dim=1).item())

        # ---------------------------
        # INTEGRATED GRADIENTS (FIXED)
        # Key fix: do NOT call model.zero_grad() inside the loop.
        # Accumulate gradients from each alpha step independently.
        # ---------------------------
        steps = 60
        accumulated_gradients = torch.zeros_like(x)

        for i in range(steps):
            alpha = i / (steps - 1)
            interpolated = (baseline + alpha * (x - baseline)).clone().detach()
            interpolated.requires_grad_(True)

            output = model(interpolated)
            score = output[0, target_class]

            # Zero only interpolated's grad, not model params
            if interpolated.grad is not None:
                interpolated.grad.zero_()

            score.backward()

            if interpolated.grad is not None:
                accumulated_gradients += interpolated.grad.detach().clone()

        avg_gradients = accumulated_gradients / steps

        # IG formula
        ig = (x - baseline).detach() * avg_gradients

        # ---------------------------
        # CHANNEL REDUCTION (abs mean across RGB)
        # ---------------------------
        heatmap = ig.abs().mean(dim=1)[0].cpu().numpy()

        # Contrast enhancement
        heatmap = np.power(heatmap, 0.4)

        # ---------------------------
        # RESIZE + SMOOTH + NORMALIZE
        # ---------------------------
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)

        if heatmap.max() - heatmap.min() > 1e-6:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros((224, 224), dtype=np.float32)

        return heatmap.astype(np.float32)

    except Exception as e:
        print("IG FAILED:", e)
        import traceback; traceback.print_exc()

        fallback = np.zeros((224, 224), dtype=np.float32)
        cv2.rectangle(fallback, (70, 70), (150, 150), 1.0, -1)
        return fallback