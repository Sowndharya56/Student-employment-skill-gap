"""Model evaluation utilities and metrics computation."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        target_classes: Optional list of class names.
        
    Returns:
        Dict: Dictionary of accuracy, precision, recall, f1 scores, and confusion matrix.
    """
    if target_classes is None:
        target_classes = sorted(list(set(y_true)))
        
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    prec_weighted = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=target_classes).tolist()
    report_dict = classification_report(y_true, y_pred, target_names=target_classes, output_dict=True, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classes": target_classes,
        "classification_report": report_dict
    }
