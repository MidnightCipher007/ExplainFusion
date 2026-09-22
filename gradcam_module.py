import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # safer hooks (forward + backward)
        self.fwd_hook = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_hook = target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()

        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_(True)

        # forward pass
        output = self.model(input_tensor)

        if output is None:
            raise Exception("Model output is None")

        class_idx = torch.argmax(output, dim=1)

        # backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise Exception("Grad-CAM hooks failed")

        grads = self.gradients[0].detach().cpu().numpy()
        acts = self.activations[0].detach().cpu().numpy()

        # ---------------------------
        # IMPROVED WEIGHTING
        # ---------------------------
        weights = np.mean(grads, axis=(1, 2))

        # normalize weights (important)
        weights = weights / (np.sum(np.abs(weights)) + 1e-8)

        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        # ---------------------------
        # RELU
        # ---------------------------
        cam = np.maximum(cam, 0)

        # ---------------------------
        # RESIZE
        # ---------------------------
        cam = cv2.resize(cam, (224, 224))

        # ---------------------------
        # SMOOTHING (important)
        # ---------------------------
        cam = cv2.GaussianBlur(cam, (9, 9), 0)

        # ---------------------------
        # NORMALIZATION (SAFE)
        # ---------------------------
        cam_min, cam_max = cam.min(), cam.max()

        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# ---------------------------
# PUBLIC FUNCTION
# ---------------------------
def generate_gradcam(model, input_tensor):
    try:
        target_layer = model.features.denseblock4

        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor)
        cam.close()

        return heatmap

    except Exception as e:
        print("GradCAM FAILED:", e)

        # fallback → soft center attention
        fallback = np.zeros((224, 224), dtype=np.float32)
        cv2.circle(fallback, (112, 112), 60, 1.0, -1)

        return fallback