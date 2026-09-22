import numpy as np
import cv2
from PIL import Image


def preprocess_medical(img):
    """
    Medical preprocessing: CLAHE + normalize.
    IMPORTANT: This must be called during BOTH training and inference
    for consistency. Apply this BEFORE the torchvision transform.
    """
    img = np.array(img)

    # ---------------------------
    # HANDLE DIFFERENT INPUT TYPES
    # ---------------------------
    if len(img.shape) == 2:
        gray = img
    elif img.shape[2] == 4:
        # RGBA → RGB → gray
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---------------------------
    # RESIZE FIRST (before CLAHE)
    # ---------------------------
    gray = cv2.resize(gray, (224, 224))

    # ---------------------------
    # CLAHE (contrast enhancement for X-rays)
    # ---------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ---------------------------
    # 3-CHANNEL CONVERSION (needed for ImageNet norm)
    # ---------------------------
    img_3ch = np.stack([gray] * 3, axis=-1)

    return Image.fromarray(img_3ch)