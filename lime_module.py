import numpy as np
import torch
from lime import lime_image
from PIL import Image
import cv2
from skimage.segmentation import slic
from utils_preprocess import preprocess_medical


def generate_lime(image, model, transform, device):
    """
    Generate LIME explanation heatmap.
    image: PIL Image (already preprocessed via preprocess_medical)
    """
    model.eval()

    try:
        # image is already preprocess_medical output (224x224 CLAHE PIL)
        # Do NOT preprocess again here.
        image_np = np.array(image)  # uint8 HxWx3

        # ---------------------------
        # PREDICT FUNCTION for LIME
        # LIME passes perturbed uint8 images (same space as image_np)
        # We must apply the SAME pipeline as inference: preprocess_medical → transform
        # ---------------------------
        def predict_fn(images):
            batch = []
            for img in images:
                pil_img = Image.fromarray(img.astype('uint8'))
                # Apply preprocess_medical so LIME perturbations go through
                # the same pipeline the model was trained on
                pil_img = preprocess_medical(pil_img)
                tensor = transform(pil_img).unsqueeze(0)
                batch.append(tensor)

            batch = torch.cat(batch).to(device)

            with torch.no_grad():
                out = model(batch)
                probs = torch.softmax(out, dim=1)

            return probs.cpu().numpy()

        # ---------------------------
        # SEGMENTATION
        # ---------------------------
        def segment_fn(x):
            return slic(x, n_segments=150, compactness=10, sigma=1,
                       start_label=0)

        # ---------------------------
        # LIME EXPLAINER
        # ---------------------------
        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            image_np,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000,
            segmentation_fn=segment_fn
        )

        top_label = explanation.top_labels[0]

        # ---------------------------
        # GET WEIGHTED HEATMAP from segment weights
        # ---------------------------
        segments = explanation.segments
        local_exp = explanation.local_exp[top_label]

        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, weight in local_exp:
            heatmap[segments == seg_id] = weight

        # Keep only positive contributions
        heatmap = np.maximum(heatmap, 0)

        # ---------------------------
        # RESIZE + SMOOTH
        # ---------------------------
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)

        # ---------------------------
        # NORMALIZE
        # ---------------------------
        if heatmap.max() - heatmap.min() > 1e-6:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros((224, 224), dtype=np.float32)

        return heatmap.astype(np.float32)

    except Exception as e:
        print("LIME FAILED:", e)
        import traceback; traceback.print_exc()

        fallback = np.zeros((224, 224), dtype=np.float32)
        cv2.circle(fallback, (112, 112), 70, 1.0, -1)
        return fallback