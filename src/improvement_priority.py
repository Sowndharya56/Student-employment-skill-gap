"""Student improvement priority calculation module."""
import os
import json
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.feature_importance import load_saved_feature_importance


def load_benchmark_statistics(stats_path: str = "data/processed/student_features_stats.json") -> Dict[str, Dict[str, float]]:
    """Load precalculated student feature distribution statistics."""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Benchmark stats file not found at {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_percentile_rank(value: float, min_val: float, max_val: float, median_val: float) -> float:
    """Approximate percentile rank assuming piecewise linear distribution between min, median, and max."""
    if max_val <= min_val:
        return 0.5
    if value <= min_val:
        return 0.01
    if value >= max_val:
        return 0.99
    if value <= median_val:
        return 0.01 + 0.49 * (value - min_val) / (median_val - min_val) if median_val > min_val else 0.5
    else:
        return 0.50 + 0.49 * (value - median_val) / (max_val - median_val) if max_val > median_val else 0.5


def calculate_improvement_priorities(
    student_dict: Dict[str, Any],
    stats_path: str = "data/processed/student_features_stats.json",
    importance_path: str = "models/feature_importance.json"
) -> List[Dict[str, Any]]:
    """Compute transparent improvement priority scores for all student attributes.
    
    Formula:
      1. Weakness Gap (W_i) = max(0, (Median - StudentValue) / (Max - Min))
      2. Percentile Rank (PR_i) = calculated position in [0, 1]
      3. Importance Weight (I_i) = normalized model feature importance
      4. Combined Gap = 0.5 * W_i + 0.5 * max(0, 0.5 - PR_i) * 2
      5. Priority Score = 100 * Combined Gap * (I_i / max(Importance))
      
    Args:
        student_dict: Dictionary of student features and values.
        stats_path: Path to dataset benchmark statistics.
        importance_path: Path to feature importance JSON.
        
    Returns:
        List of priority records sorted descending by Priority Score.
    """
    stats = load_benchmark_statistics(stats_path)
    importances = load_saved_feature_importance(importance_path)
    max_imp = max(importances.values()) if importances else 1.0
    
    priorities = []
    
    for feature, stat in stats.items():
        if feature not in student_dict:
            continue
            
        val = float(student_dict[feature])
        min_val = stat["min"]
        max_val = stat["max"]
        med_val = stat["median"]
        imp_val = importances.get(feature, 1.0 / len(stats))
        
        # 1. Weakness gap relative to range
        rng = max_val - min_val if max_val > min_val else 1.0
        weakness_gap = max(0.0, (med_val - val) / rng)
        
        # 2. Percentile rank
        pct_rank = calculate_percentile_rank(val, min_val, max_val, med_val)
        
        # 3. Position category
        if val < med_val:
            pos_label = f"Below Median ({int(pct_rank * 100)}th percentile)"
        elif val == med_val:
            pos_label = f"At Median (50th percentile)"
        else:
            pos_label = f"Above Median ({int(pct_rank * 100)}th percentile)"
            
        # 4. Importance category
        if imp_val >= 0.10:
            imp_label = "High"
        elif imp_val >= 0.06:
            imp_label = "Moderate"
        else:
            imp_label = "Normal"
            
        # 5. Combined Priority Score (0-100)
        percentile_gap = max(0.0, 0.5 - pct_rank) * 2.0
        combined_gap = 0.5 * weakness_gap + 0.5 * percentile_gap
        priority_score = min(100.0, combined_gap * (imp_val / max_imp) * 100.0)
        
        # Human-readable reason
        feature_clean_name = feature.replace('_', ' ')
        if val < med_val:
            reason = f"{imp_label} model importance ({imp_val*100:.1f}%) and current value ({val:g}) is below dataset median ({med_val:g})."
        elif val == med_val:
            reason = f"{imp_label} model importance ({imp_val*100:.1f}%), performing at the cohort median."
        else:
            reason = f"Performing above cohort median ({val:g} vs {med_val:g}); maintain current performance."
            
        priorities.append({
            "feature": feature,
            "feature_display": feature_clean_name,
            "current_value": val,
            "dataset_median": med_val,
            "dataset_min": min_val,
            "dataset_max": max_val,
            "dataset_position": pos_label,
            "model_importance_pct": imp_val * 100,
            "importance_level": imp_label,
            "priority_score": round(priority_score, 1),
            "reason": reason
        })
        
    # Sort descending by priority score
    priorities.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # Assign 1-indexed ranks
    for idx, item in enumerate(priorities, 1):
        item["rank"] = idx
        
    return priorities
