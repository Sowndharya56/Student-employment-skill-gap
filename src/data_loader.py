"""Data loading and schema validation module."""
import os
import pandas as pd
from typing import Tuple, Dict, Any, Optional

EXPECTED_STUDENT_COLUMNS = [
    'GPA',
    'Credits_Completed',
    'Exam_Scores',
    'Attendance_Percentage',
    'Coding_Skills',
    'Communication_Skills',
    'Leadership_Skills',
    'Clubs_Joined',
    'Events_Participated',
    'Internship_Completed',
    'Internship_Duration_Months',
    'Library_Visits_Per_Week',
    'Online_Course_Hours_Per_Week',
    'Employment_Competitiveness'
]

EXPECTED_JOB_COLUMNS = [
    'Job Title',
    'Job Description',
    'Skills',
    'Certifications'
]


def load_student_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and validate the student employment competitiveness dataset.
    
    Args:
        filepath: Optional custom filepath. Defaults to 'data/raw/College_Employment_Competitiveness.csv'
                  or 'College_Employment_Competitiveness.csv'.
    
    Returns:
        pd.DataFrame: Loaded student dataset.
    """
    if filepath is None:
        candidates = [
            os.path.join('data', 'raw', 'College_Employment_Competitiveness.csv'),
            'College_Employment_Competitiveness.csv'
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
    
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Student dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Schema validation
    missing_cols = [c for c in EXPECTED_STUDENT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Student dataset missing expected columns: {missing_cols}")
    
    return df


def load_job_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load and validate the IT Job Roles and Skills dataset.
    
    Args:
        filepath: Optional custom filepath. Defaults to 'data/raw/IT_Job_Roles_Skills.csv'
                  or 'IT_Job_Roles_Skills.csv'.
    
    Returns:
        pd.DataFrame: Loaded IT job roles dataset.
    """
    if filepath is None:
        candidates = [
            os.path.join('data', 'raw', 'IT_Job_Roles_Skills.csv'),
            'IT_Job_Roles_Skills.csv'
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = c
                break
    
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Job dataset not found at {filepath}")
    
    # Try cp1252, latin1, utf-8
    df = None
    for enc in ['cp1252', 'latin1', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception:
            continue
            
    if df is None:
        raise ValueError(f"Could not read job dataset with supported encodings at {filepath}")
    
    # Schema validation
    missing_cols = [c for c in EXPECTED_JOB_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Job dataset missing expected columns: {missing_cols}")
    
    return df


def get_dataset_metadata() -> Dict[str, Any]:
    """Retrieve high-level metadata summary for both datasets."""
    student_df = load_student_dataset()
    job_df = load_job_dataset()
    
    return {
        "student_dataset": {
            "rows": len(student_df),
            "columns": list(student_df.columns),
            "target": "Employment_Competitiveness",
            "target_classes": list(student_df["Employment_Competitiveness"].unique()),
            "missing_values": int(student_df.isnull().sum().sum()),
            "duplicates": int(student_df.duplicated().sum())
        },
        "job_dataset": {
            "rows": len(job_df),
            "columns": list(job_df.columns),
            "unique_roles": int(job_df["Job Title"].nunique()),
            "missing_values": int(job_df.isnull().sum().sum()),
            "duplicates": int(job_df.duplicated().sum())
        }
    }
