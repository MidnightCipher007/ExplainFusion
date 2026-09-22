import gradio as gr
import numpy as np
import traceback

from model import load_model, predict, transform, device
from gradcam_module import generate_gradcam
from lime_module import generate_lime
from shap_module import generate_shap
from fusion_module import fuse_heatmaps, compute_agreement
from utils import overlay_heatmap
from llm_module import generate_explanation

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model()


# ---------------------------
# SAFE NORMALIZATION
# ---------------------------
def safe_norm(x):
    if x is None:
        return np.zeros((224, 224), dtype=np.float32)

    x = np.nan_to_num(x)

    if x.max() - x.min() < 1e-6:
        return np.zeros_like(x)

    return (x - x.min()) / (x.max() - x.min())


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def run_explainfusion(image):
    try:
        if image is None:
            return "⚠️ Please upload an image.", None, None, None, None

        # ---------------------------
        # PREDICTION
        # ---------------------------
        label, confidence, input_tensor, probs, processed = predict(model, image)

        # ---------------------------
        # GRADCAM
        # ---------------------------
        gradcam = safe_norm(generate_gradcam(model, input_tensor))

        # ---------------------------
        # LIME
        # ---------------------------
        try:
            lime = safe_norm(generate_lime(processed, model, transform, device))
        except:
            lime = np.zeros((224, 224))

        # ---------------------------
        # SHAP
        # ---------------------------
        try:
            shap = safe_norm(generate_shap(processed, model, transform, device))
        except:
            shap = gradcam.copy()

        # ---------------------------
        # FUSION
        # ---------------------------
        fused = safe_norm(fuse_heatmaps(gradcam, lime, shap))

        # ---------------------------
        # AGREEMENT
        # ---------------------------
        agreement = compute_agreement(gradcam, lime, shap)

        # ---------------------------
        # TEXT OUTPUT
        # ---------------------------
        explanation = generate_explanation(label, confidence, agreement)

        # Add probability display
        prob_text = f"\n\n🔢 Probabilities:\nNormal: {probs[0]:.2f}\nPneumonia: {probs[1]:.2f}"

        explanation = explanation + prob_text

        # ---------------------------
        # VISUALS
        # ---------------------------
        grad_img = overlay_heatmap(processed, gradcam)
        lime_img = overlay_heatmap(processed, lime)
        shap_img = overlay_heatmap(processed, shap)
        fused_img = overlay_heatmap(processed, fused, alpha=0.6)

        return explanation, grad_img, lime_img, shap_img, fused_img

    except Exception as e:
        print("APP ERROR:")
        traceback.print_exc()

        return f"❌ Error: {str(e)}", None, None, None, None


# ---------------------------
# CLEAR FUNCTION (FIXED)
# ---------------------------
def clear_all():
    return None, "", None, None, None, None


# ---------------------------
# UI STYLING
# ---------------------------
css = """
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: Arial;
}

h1 {
    text-align: center;
    font-size: 30px;
}

.gr-button {
    font-size: 16px;
}
"""


# ---------------------------
# UI LAYOUT
# ---------------------------
with gr.Blocks(css=css) as demo:

    gr.Markdown("# 🧠 ExplainFusion: Explainable Medical AI")
    gr.Markdown("Upload a chest X-ray to detect pneumonia and visualize AI reasoning.")

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload X-ray")

            analyze_btn = gr.Button("🔍 Analyze", variant="primary")
            clear_btn = gr.Button("🧹 Clear")

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="AI Diagnostic Report",
                lines=22
            )

    gr.Markdown("## 🔬 Explainability Maps")

    with gr.Row():
        grad_out = gr.Image(label="Grad-CAM")
        lime_out = gr.Image(label="LIME")
        shap_out = gr.Image(label="SHAP (IG)")
        fused_out = gr.Image(label="Fused")

    # ---------------------------
    # ACTIONS
    # ---------------------------
    analyze_btn.click(
        fn=run_explainfusion,
        inputs=input_img,
        outputs=[output_text, grad_out, lime_out, shap_out, fused_out]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_img, output_text, grad_out, lime_out, shap_out, fused_out]
    )

# ---------------------------
# RUN
# ---------------------------
demo.launch(share=True)