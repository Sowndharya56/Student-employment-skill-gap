"""Model training, cross-validation comparison, and model serialization module."""
import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

from src.data_cleaning import process_and_save_data
from src.preprocessing import split_data, FEATURE_COLUMNS, TARGET_CLASSES
from src.model_evaluation import evaluate_classifier


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classification models wrapped in pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7))
        ])
    }


def compute_feature_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute feature importance scores either natively or via permutation importance on test data.
    
    Args:
        pipeline: Trained scikit-learn pipeline.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dict: Mapping of feature names to normalized importance scores (summing to 1.0).
    """
    classifier = pipeline.named_steps.get("classifier", pipeline)
    importances = {}
    
    # 1. Check tree-based feature importances
    if hasattr(classifier, "feature_importances_"):
        raw_imp = classifier.feature_importances_
        for col, imp in zip(FEATURE_COLUMNS, raw_imp):
            importances[col] = float(imp)
            
    # 2. Check linear model coefficients (mean absolute across classes)
    elif hasattr(classifier, "coef_"):
        raw_imp = np.mean(np.abs(classifier.coef_), axis=0)
        for col, imp in zip(FEATURE_COLUMNS, raw_imp):
            importances[col] = float(imp)
            
    # 3. Fallback to permutation importance
    else:
        perm = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, scoring='accuracy')
        raw_imp = np.maximum(0, perm.importances_mean)
        # In case all are 0, assign uniform weights
        if np.sum(raw_imp) == 0:
            raw_imp = np.ones(len(FEATURE_COLUMNS))
        for col, imp in zip(FEATURE_COLUMNS, raw_imp):
            importances[col] = float(imp)
            
    # Normalize so sum is 1.0
    total = sum(importances.values())
    if total > 0:
        importances = {k: float(v / total) for k, v in importances.items()}
    else:
        importances = {k: float(1.0 / len(FEATURE_COLUMNS)) for k in FEATURE_COLUMNS}
        
    # Sort descending
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def train_and_evaluate_all_models(
    models_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """Run cross-validation comparison, select the best model, evaluate on test set,
    and serialize all artifacts to disk.
    
    Args:
        models_dir: Directory to save serialized model and metadata.
        random_state: Seed for splits and models.
        
    Returns:
        Dict containing training and comparison results.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and clean data
    cleaned_student, _, _ = process_and_save_data()
    X_train, X_test, y_train, y_test = split_data(cleaned_student, test_size=0.2, random_state=random_state)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    candidate_models = get_candidate_models()
    
    comparison_results = {}
    best_model_name = None
    best_val_f1 = -1.0
    
    for name, pipeline in candidate_models.items():
        # Cross-validation metrics
        cv_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro')
        cv_acc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        mean_f1 = float(np.mean(cv_f1_scores))
        std_f1 = float(np.std(cv_f1_scores))
        mean_acc = float(np.mean(cv_acc_scores))
        std_acc = float(np.std(cv_acc_scores))
        
        # Fit on entire training set
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)
        
        test_eval = evaluate_classifier(y_test, y_test_pred, target_classes=TARGET_CLASSES)
        
        comparison_results[name] = {
            "cv_macro_f1_mean": mean_f1,
            "cv_macro_f1_std": std_f1,
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": std_acc,
            "test_accuracy": test_eval["accuracy"],
            "test_macro_f1": test_eval["f1_macro"],
            "test_weighted_f1": test_eval["f1_weighted"],
            "test_evaluation": test_eval
        }
        
        # Validation F1 selection
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            best_model_name = name
            
    # Best model selection & final fitting
    best_pipeline = candidate_models[best_model_name]
    best_pipeline.fit(X_train, y_train)
    
    # Feature importance
    feat_importance = compute_feature_importance(best_pipeline, X_test, y_test)
    
    # Save best model
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)
    
    # Save comparison report
    with open(os.path.join(models_dir, "all_models_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
        
    # Save feature importances
    with open(os.path.join(models_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feat_importance, f, indent=2)
        
    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "selected_by": "5-Fold Stratified CV Macro F1",
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_dataset_rows": len(cleaned_student),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "Employment_Competitiveness",
        "target_classes": TARGET_CLASSES,
        "cv_macro_f1_mean": comparison_results[best_model_name]["cv_macro_f1_mean"],
        "cv_accuracy_mean": comparison_results[best_model_name]["cv_accuracy_mean"],
        "test_evaluation": comparison_results[best_model_name]["test_evaluation"],
        "disclaimer": "Model performance is measured on this dataset and does not guarantee real-world employment outcomes."
    }
    
    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    return {
        "best_model_name": best_model_name,
        "comparison_results": comparison_results,
        "metadata": metadata,
        "feature_importance": feat_importance
    }


if __name__ == "__main__":
    results = train_and_evaluate_all_models()
    print(f"Model training complete. Selected Best Model: {results['best_model_name']}")
    print(f"Validation Macro F1: {results['metadata']['cv_macro_f1_mean']:.4f}")
    print(f"Test Accuracy: {results['metadata']['test_evaluation']['accuracy']:.4f}")
    print(f"Test Macro F1: {results['metadata']['test_evaluation']['f1_macro']:.4f}")
