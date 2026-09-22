import cv2
import numpy as np

def overlay_heatmap(image, heatmap, alpha=0.5, threshold=0.05):
    """
    Overlay heatmap on image with proper normalization and thresholding
    """

    # Convert PIL -> numpy
    img = np.array(image)

    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize to model size (IMPORTANT)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))

    # ---------------------------
    # 🔥 NORMALIZE SAFELY
    # ---------------------------
    heatmap = heatmap - np.min(heatmap)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    # ---------------------------
    # 🔥 THRESHOLD (your feature)
    # ---------------------------
    heatmap[heatmap < threshold] = 0

    # ---------------------------
    # Convert to color
    # ---------------------------
    heatmap_uint8 = np.uint8(255 * heatmap)

    # OpenCV uses BGR → convert later
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert BGR → RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # ---------------------------
    # Overlay
    # ---------------------------
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    return overlay