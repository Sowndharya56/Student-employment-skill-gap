"""Prediction and student evaluation inference module."""
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.preprocessing import FEATURE_COLUMNS, TARGET_CLASSES


class CompetitivenessPredictor:
    """Predictor class for student employment competitiveness inference."""
    
    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        metadata_path: str = "models/model_metadata.json"
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load trained pipeline and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run model training first.")
        self.pipeline = joblib.load(self.model_path)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "model_name": "Random Forest",
                "target_classes": TARGET_CLASSES,
                "disclaimer": "Model performance is measured on this dataset and does not guarantee real-world employment outcomes."
            }
            
    def predict_student(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Generate competitiveness prediction, probability distributions, and explanation.
        
        Args:
            student_dict: Dictionary containing student input attributes.
            
        Returns:
            Dict containing predicted class, confidence, probabilities, and model info.
        """
        # Validate and build 1-row DataFrame matching FEATURE_COLUMNS
        row_data = {}
        for col in FEATURE_COLUMNS:
            if col not in student_dict:
                raise ValueError(f"Missing required student feature: '{col}'")
            row_data[col] = [float(student_dict[col])]
            
        input_df = pd.DataFrame(row_data)
        
        # Predict class
        pred_class = self.pipeline.predict(input_df)[0]
        
        # Predict probabilities if supported
        probabilities = {}
        confidence = None
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(input_df)[0]
            classes = list(self.pipeline.classes_)
            for cls_name, prob_val in zip(classes, probs):
                probabilities[str(cls_name)] = float(prob_val)
            confidence = float(probabilities.get(str(pred_class), max(probs)))
            
        return {
            "predicted_competitiveness": str(pred_class),
            "confidence_score": confidence,
            "probabilities": probabilities,
            "model_name": self.metadata.get("model_name", "Classification Model"),
            "disclaimer": "Based on the trained ML model and the information provided.",
            "input_features": student_dict
        }


# Global singleton instance helper
_predictor_instance = None


def get_predictor() -> CompetitivenessPredictor:
    """Get or instantiate singleton CompetitivenessPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CompetitivenessPredictor()
    return _predictor_instance


def predict_single_student(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """Functional convenience wrapper for single student prediction."""
    predictor = get_predictor()
    return predictor.predict_student(student_data)
