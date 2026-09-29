"""Preprocessing pipeline module ensuring zero data leakage."""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

TARGET_COLUMN = 'Employment_Competitiveness'
FEATURE_COLUMNS = [
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
    'Online_Course_Hours_Per_Week'
]

# Standard target class mapping
TARGET_CLASSES = ['High', 'Medium', 'Low']
TARGET_TO_NUM = {'Low': 0, 'Medium': 1, 'High': 2}
NUM_TO_TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}


def get_feature_and_target_columns() -> Tuple[List[str], str]:
    """Return feature column list and target column name."""
    return FEATURE_COLUMNS, TARGET_COLUMN


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into stratified train and test sets without leakage.
    
    Args:
        df: Input DataFrame.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def create_preprocessor() -> ColumnTransformer:
    """Create a standard scikit-learn ColumnTransformer preprocessor."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, FEATURE_COLUMNS)
        ],
        remainder='drop'
    )
    
    return preprocessor
