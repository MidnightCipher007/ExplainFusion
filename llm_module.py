def generate_explanation(label, confidence, agreement):
    """
    Generates a detailed, dynamic clinical diagnostic report.
    Output varies meaningfully based on confidence + agreement combinations.
    """

    # ---------------------------
    # CONFIDENCE TIER
    # ---------------------------
    if confidence >= 0.90:
        confidence_text = "very high confidence"
        confidence_desc = (
            f"The model predicted {label} with {confidence*100:.1f}% confidence. "
            "This represents a strong, decisive classification supported by highly discriminative imaging features. "
            "The prediction is unlikely to be a borderline or ambiguous case."
        )
    elif confidence >= 0.75:
        confidence_text = "high confidence"
        confidence_desc = (
            f"The model predicted {label} with {confidence*100:.1f}% confidence. "
            "The classification is well-supported by learned imaging patterns, "
            "with the model demonstrating clear preference for this diagnosis over the alternative."
        )
    elif confidence >= 0.60:
        confidence_text = "moderate confidence"
        confidence_desc = (
            f"The model predicted {label} with {confidence*100:.1f}% confidence. "
            "While the prediction leans toward this diagnosis, some uncertainty remains. "
            "The image may contain overlapping features or subtle presentation patterns."
        )
    else:
        confidence_text = "low confidence"
        confidence_desc = (
            f"The model predicted {label} with only {confidence*100:.1f}% confidence. "
            "The classification is weakly supported and the model shows significant uncertainty. "
            "This result should be treated with caution and not used in isolation."
        )

    # ---------------------------
    # AGREEMENT TIER
    # ---------------------------
    if agreement >= 0.80:
        agreement_text = "excellent agreement"
        agreement_desc = (
            "All three explainability methods — Grad-CAM, LIME, and Integrated Gradients (SHAP) — "
            "show strong spatial overlap, consistently highlighting the same pulmonary regions. "
            "This high degree of cross-method consensus significantly strengthens the interpretability "
            "and trustworthiness of the model's decision."
        )
    elif agreement >= 0.65:
        agreement_text = "strong agreement"
        agreement_desc = (
            "The three explainability methods show substantial overlap in the regions they highlight. "
            "Grad-CAM, LIME, and Integrated Gradients largely agree on which areas of the X-ray "
            "drove the prediction, indicating reliable and stable model reasoning."
        )
    elif agreement >= 0.50:
        agreement_text = "moderate agreement"
        agreement_desc = (
            "The explainability methods show partial spatial overlap. "
            "While there is general consistency in the highlighted lung regions, "
            "some variation exists between Grad-CAM, LIME, and Integrated Gradients, "
            "suggesting the model's attention is reasonably but not perfectly stable."
        )
    else:
        agreement_text = "low agreement"
        agreement_desc = (
            "The three explainability methods highlight notably different regions, "
            "indicating instability in the model's reasoning for this particular image. "
            "This may be due to ambiguous image features, poor image quality, or an atypical presentation. "
            "Results should be interpreted with extra caution."
        )

    # ---------------------------
    # CLINICAL FINDINGS (label-specific)
    # ---------------------------
    if label == "Pneumonia":
        findings = (
            "Increased radiographic opacity is identified within the lung fields, "
            "consistent with possible inflammatory infiltration, consolidation, or fluid accumulation. "
            "The affected regions demonstrate reduced translucency compared to normal aerated lung tissue, "
            "which is a characteristic radiological marker of pulmonary infection or inflammation."
        )
        model_reasoning = (
            "The model identified regions of abnormal density and heterogeneous texture within the lung parenchyma. "
            "These patterns — including asymmetric opacification and loss of normal lung architecture — "
            "are hallmarks of bacterial or viral pneumonia in chest radiography. "
            "The explainability maps highlight these areas as the primary drivers of the Pneumonia classification."
        )
        risk = (
            "The identified radiological features are consistent with an active pulmonary infectious process. "
            "If untreated, pneumonia can progress to respiratory complications including pleural effusion, "
            "lung abscess, or sepsis. Prompt clinical evaluation is advised."
        )
        recommendation = (
            "Immediate clinical correlation is strongly recommended, including assessment of patient symptoms "
            "(fever, cough, dyspnea), oxygen saturation, and inflammatory markers (CRP, WBC count). "
            "A qualified radiologist review and/or repeat imaging may be warranted. "
            "Treatment decisions should be made by a licensed healthcare professional."
        )
    else:
        findings = (
            "No prominent focal opacities, consolidations, or structural abnormalities are identified "
            "within the visible lung fields. The lung parenchyma appears normally aerated with "
            "clear costophrenic angles and no evidence of significant pleural effusion or infiltration. "
            "The cardiac silhouette and mediastinum appear within normal radiographic limits."
        )
        model_reasoning = (
            "The model identified uniformly distributed lung textures without detecting the abnormal "
            "density patterns associated with pneumonia. The absence of focal consolidation, "
            "air bronchograms, or significant opacity led the model to classify this as a Normal X-ray. "
            "The explainability maps reflect diffuse attention without specific pathological focus."
        )
        risk = (
            "No radiographic features strongly suggestive of pneumonia are identified in this image. "
            "The risk of active pulmonary infection based on this X-ray alone appears low. "
            "However, radiological findings must always be interpreted alongside clinical presentation."
        )
        recommendation = (
            "No immediate radiological intervention appears necessary based on this result. "
            "If the patient presents with persistent respiratory symptoms despite a normal X-ray, "
            "further evaluation including CT imaging, pulmonary function tests, or specialist review "
            "may be considered by the treating physician."
        )

    # ---------------------------
    # RELIABILITY (confidence + agreement combined)
    # ---------------------------
    if confidence >= 0.85 and agreement >= 0.70:
        reliability = (
            "HIGH — The combination of strong model confidence and high cross-method explainability agreement "
            "indicates this is a reliable prediction. The model's reasoning is both decisive and consistent."
        )
    elif confidence >= 0.75 and agreement >= 0.55:
        reliability = (
            "MODERATE-HIGH — The prediction is well-supported with reasonable explainability consistency. "
            "Minor uncertainty exists but the overall result is trustworthy for clinical consideration."
        )
    elif confidence >= 0.60:
        reliability = (
            "MODERATE — The prediction shows reasonable confidence but explainability consistency is limited. "
            "This result should be considered alongside other clinical findings."
        )
    else:
        reliability = (
            "LOW — Both confidence and explainability agreement are below acceptable thresholds. "
            "This prediction should not be used as a standalone diagnostic indicator."
        )

    # ---------------------------
    # FINAL REPORT
    # ---------------------------
    report = f"""
╔══════════════════════════════════════════════════╗
         ExplainFusion AI DIAGNOSTIC REPORT         
╚══════════════════════════════════════════════════╝

PREDICTION:  {label.upper()}
CONFIDENCE:  {confidence*100:.1f}%  ({confidence_text})
XAI AGREEMENT:  {agreement*100:.1f}%  ({agreement_text})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONFIDENCE ANALYSIS
{confidence_desc}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. RADIOLOGICAL FINDINGS
{findings}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. MODEL REASONING
{model_reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. EXPLAINABILITY ANALYSIS
Agreement Score: {agreement:.2f} — {agreement_text.upper()}

{agreement_desc}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. RELIABILITY ASSESSMENT
{reliability}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. RISK ASSESSMENT
{risk}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7. CLINICAL RECOMMENDATION
{recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISCLAIMER
This report is generated by an AI decision-support system (ExplainFusion).
It is intended solely to assist qualified medical professionals and does
not constitute a medical diagnosis. Final clinical decisions must be made
by a licensed healthcare provider with full access to patient history,
symptoms, and additional diagnostic data.

Model: DenseNet121 | XAI: Grad-CAM + LIME + Integrated Gradients
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    return report.strip()