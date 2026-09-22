# ExplainFusion

An Ensemble Explainable AI Framework with Evidence-Grounded
LLM-Based Clinical Explanations

## Overview

ExplainFusion is an explainable AI framework designed for
chest X-ray analysis. The system combines a DenseNet-based
deep learning classifier with multiple explainability
techniques and evidence-grounded language generation.

The framework provides:

1. Disease prediction
2. Prediction confidence
3. Grad-CAM explanation
4. LIME explanation
5. SHAP explanation
6. Fused explanation heatmap
7. Structured evidence extraction
8. Evidence-grounded textual explanation

## System Architecture

Chest X-ray
      ↓
DenseNet Classifier
      ↓
Prediction + Confidence
      ↓
Grad-CAM + LIME + SHAP
      ↓
Heatmap Normalization
      ↓
Explanation Fusion
      ↓
Structured Evidence
      ↓
LLM Explanation
      ↓
Final Visual + Textual Explanation

## Technologies

- Python
- PyTorch
- DenseNet121
- Grad-CAM
- LIME
- SHAP
- NumPy
- OpenCV / PIL
- Matplotlib
- Web interface


## Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL

cd ExplainFusion

Install dependencies:

pip install -r requirements.txt

## Running the Project

Run:

python app.py

Then open the local web interface displayed by the application.

## Evaluation

The framework can be evaluated using:

- Intersection over Union (IoU)
- Pointing Accuracy
- Faithfulness
- Qualitative explanation analysis

## Project Structure

src/
    model.py
    gradcam_module.py
    lime_module.py
    shap_module.py
    fusion_module.py
    llm_module.py
    utils.py

results/
    Sample outputs and evaluation results

## Dataset

The project uses the RSNA Pneumonia Detection Challenge chest X-ray dataset, obtained through Kaggle.

The dataset contains chest X-ray images and pneumonia-related annotations used for the classification and explainability pipeline.

The complete dataset is not included in this repository because of its size and applicable dataset usage requirements.

### Dataset Setup

1. Download the RSNA Pneumonia Detection Challenge dataset from Kaggle.
2. Extract the downloaded files locally.
3. Configure the dataset path according to the training and preprocessing scripts.
4. Run the preprocessing/training pipeline as required.

The dataset itself is not included in this repository.

## Limitations

This project is a research prototype and has not undergone
prospective clinical validation with practicing radiologists.
The computational cost of multiple XAI techniques can also
affect real-time deployment.

## Future Work

- Clinical validation
- Learning-based explanation fusion
- Support for CT and MRI
- Computational optimization
- Improved evidence-grounded generation

Author: 
ASHISH SINGH
