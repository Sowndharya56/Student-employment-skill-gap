"""Tests for model training, evaluation, inference, and feature importance."""
import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_student_dataset
from src.preprocessing import split_data, FEATURE_COLUMNS, TARGET_CLASSES
from src.prediction import get_predictor, predict_single_student
from src.feature_importance import load_saved_feature_importance, get_top_contributing_factors


def test_data_split_no_leakage():
    df = load_student_dataset()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) == 3200
    assert len(X_test) == 800
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert "Employment_Competitiveness" not in X_train.columns
    # Check stratification
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    assert abs(train_dist['High'] - 0.333) < 0.01
    assert abs(test_dist['High'] - 0.333) < 0.01


def test_predictor_inference():
    predictor = get_predictor()
    sample_student = {
        'GPA': 3.5,
        'Credits_Completed': 100,
        'Exam_Scores': 85,
        'Attendance_Percentage': 90.0,
        'Coding_Skills': 8,
        'Communication_Skills': 7,
        'Leadership_Skills': 6,
        'Clubs_Joined': 3,
        'Events_Participated': 6,
        'Internship_Completed': 1,
        'Internship_Duration_Months': 8,
        'Library_Visits_Per_Week': 10,
        'Online_Course_Hours_Per_Week': 14
    }
    result = predictor.predict_student(sample_student)
    assert "predicted_competitiveness" in result
    assert result["predicted_competitiveness"] in TARGET_CLASSES
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert len(result["probabilities"]) == 3
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4


def test_feature_importance_loaded():
    importances = load_saved_feature_importance()
    assert len(importances) == len(FEATURE_COLUMNS)
    assert abs(sum(importances.values()) - 1.0) < 1e-4
    top_5 = get_top_contributing_factors(5)
    assert len(top_5) == 5
    for feat, score in top_5:
        assert feat in FEATURE_COLUMNS
        assert score > 0
