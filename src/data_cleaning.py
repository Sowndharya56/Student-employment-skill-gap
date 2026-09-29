"""Data cleaning and statistical summary module."""
import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.data_loader import load_student_dataset, load_job_dataset


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean student dataset by trimming whitespace, verifying ranges, and handling duplicates.
    
    Args:
        df: Raw student DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned = df.copy()
    
    # Drop exact duplicates if any
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    
    # Clean target string whitespace
    if 'Employment_Competitiveness' in cleaned.columns:
        cleaned['Employment_Competitiveness'] = cleaned['Employment_Competitiveness'].astype(str).str.strip()
    
    return cleaned


def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean IT job dataset by trimming whitespace and normalizing text.
    
    Args:
        df: Raw job DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned = df.copy()
    
    for col in ['Job Title', 'Job Description', 'Skills', 'Certifications']:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()
    
    # Drop exact duplicate rows
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    
    return cleaned


def compute_student_benchmark_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute benchmark statistics (min, max, mean, std, median, 25th, 75th percentiles)
    for all numerical student features.
    
    Args:
        df: Student DataFrame.
        
    Returns:
        Dict: Mapping of feature name to its distribution statistics.
    """
    feature_cols = [c for c in df.columns if c != 'Employment_Competitiveness']
    stats = {}
    
    for col in feature_cols:
        series = df[col].astype(float)
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25))
        }
        
    return stats


def process_and_save_data(output_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load, clean, compute statistics, and save processed artifacts.
    
    Args:
        output_dir: Directory to save processed files.
        
    Returns:
        Tuple of (cleaned_student_df, cleaned_job_df, student_stats)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    student_raw = load_student_dataset()
    job_raw = load_job_dataset()
    
    cleaned_student = clean_student_data(student_raw)
    cleaned_job = clean_job_data(job_raw)
    student_stats = compute_student_benchmark_stats(cleaned_student)
    
    # Save processed CSVs
    cleaned_student.to_csv(os.path.join(output_dir, "cleaned_student_data.csv"), index=False)
    cleaned_job.to_csv(os.path.join(output_dir, "cleaned_it_job_roles.csv"), index=False)
    
    # Save feature statistics as JSON
    with open(os.path.join(output_dir, "student_features_stats.json"), "w", encoding="utf-8") as f:
        json.dump(student_stats, f, indent=2)
        
    return cleaned_student, cleaned_job, student_stats


if __name__ == "__main__":
    process_and_save_data()
    print("Data cleaning and processing completed successfully!")
