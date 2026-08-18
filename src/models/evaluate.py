"""
Model evaluation module.
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Config


def evaluate_model(model, X_test, y_test, class_labels=None):
    if class_labels is None:
        class_labels = Config.CLASS_LABELS
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=[class_labels[i] for i in sorted(class_labels.keys())])
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("="*60)
    return {'accuracy': accuracy, 'confusion_matrix': cm, 'predictions': y_pred}
