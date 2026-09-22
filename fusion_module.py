import numpy as np
import cv2


# ---------------------------
# NORMALIZATION (SAFE)
# ---------------------------
def normalize(x):
    x = x.astype(np.float32)
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x


# ---------------------------
# QUALITY SCORE (NEW 🔥)
# ---------------------------
def quality_score(x):
    """
    Measures how 'focused' a heatmap is
    """
    return float(np.mean(x > 0.5))


# ---------------------------
# FUSION (IMPROVED)
# ---------------------------
def fuse_heatmaps(gradcam, lime, shap):
    # resize + normalize
    g = normalize(cv2.resize(gradcam, (224, 224)))
    l = normalize(cv2.resize(lime, (224, 224)))
    s = normalize(cv2.resize(shap, (224, 224)))

    # ---------------------------
    # 🔥 REMOVE NOISE
    # ---------------------------
    g[g < 0.2] = 0
    l[l < 0.2] = 0
    s[s < 0.2] = 0

    # ---------------------------
    # 🔥 ADAPTIVE WEIGHTS
    # ---------------------------
    wg = quality_score(g)
    wl = quality_score(l)
    ws = quality_score(s)

    total = wg + wl + ws + 1e-8

    wg /= total
    wl /= total
    ws /= total

    # fallback (if all weak)
    if total < 1e-6:
        wg, wl, ws = 0.5, 0.3, 0.2

    # ---------------------------
    # FUSION
    # ---------------------------
    fused = wg * g + wl * l + ws * s

    # ---------------------------
    # 🔥 SHARPEN (important)
    # ---------------------------
    fused = np.power(fused, 0.7)

    # smooth slightly
    fused = cv2.GaussianBlur(fused, (7, 7), 0)

    return normalize(fused)


# ---------------------------
# AGREEMENT (IMPROVED)
# ---------------------------
def compute_agreement(g, l, s):
    g, l, s = normalize(g), normalize(l), normalize(s)

    # pairwise similarity
    diff_gl = np.mean(np.abs(g - l))
    diff_ls = np.mean(np.abs(l - s))
    diff_sg = np.mean(np.abs(s - g))

    diff = (diff_gl + diff_ls + diff_sg) / 3

    # convert to agreement score
    agreement = 1 - diff

    # clamp
    return float(max(0.0, min(1.0, agreement)))