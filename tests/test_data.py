"""Tests for data loading, schema validation, and cleaning."""
import pytest
import pandas as pd
from src.data_loader import (
    load_student_dataset,
    load_job_dataset,
    get_dataset_metadata,
    EXPECTED_STUDENT_COLUMNS,
    EXPECTED_JOB_COLUMNS
)
from src.data_cleaning import clean_student_data, clean_job_data, compute_student_benchmark_stats


def test_load_student_dataset():
    df = load_student_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4000
    assert list(df.columns) == EXPECTED_STUDENT_COLUMNS
    assert df.isnull().sum().sum() == 0
    assert set(df['Employment_Competitiveness'].unique()) == {'High', 'Medium', 'Low'}


def test_load_job_dataset():
    df = load_job_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 490
    for col in EXPECTED_JOB_COLUMNS:
        assert col in df.columns
    assert df['Job Title'].nunique() >= 460


def test_get_dataset_metadata():
    meta = get_dataset_metadata()
    assert "student_dataset" in meta
    assert "job_dataset" in meta
    assert meta["student_dataset"]["rows"] == 4000
    assert meta["student_dataset"]["target"] == "Employment_Competitiveness"
    assert meta["job_dataset"]["unique_roles"] >= 460


def test_clean_student_data():
    df = load_student_dataset()
    cleaned = clean_student_data(df)
    assert len(cleaned) == 4000
    assert cleaned.duplicated().sum() == 0


def test_clean_job_data():
    df = load_job_dataset()
    cleaned = clean_job_data(df)
    assert len(cleaned) <= len(df)
    assert cleaned.duplicated().sum() == 0


def test_compute_student_benchmark_stats():
    df = load_student_dataset()
    stats = compute_student_benchmark_stats(df)
    assert "GPA" in stats
    assert "Coding_Skills" in stats
    assert stats["GPA"]["min"] == 2.0
    assert stats["GPA"]["max"] == 4.0
    assert 2.9 <= stats["GPA"]["median"] <= 3.1
