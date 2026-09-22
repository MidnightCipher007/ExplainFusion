import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate(model, loader, device, class_names=("Normal", "Pneumonia")):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---------------------------
    # METRICS
    # ---------------------------
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # ---------------------------
    # PRINT RESULTS
    # ---------------------------
    print("\n================ MODEL EVALUATION ================\n")

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n---------------- Confusion Matrix ----------------")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n------------- Classification Report -------------")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\n=================================================\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }