"""Feature importance extraction and visualization module."""
import os
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd


def load_saved_feature_importance(json_path: str = "models/feature_importance.json") -> Dict[str, float]:
    """Load calculated feature importance from JSON file.
    
    Args:
        json_path: Path to the feature importance JSON file.
        
    Returns:
        Dict: Mapping of feature names to relative importance weights.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Feature importance file not found at {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_top_contributing_factors(top_n: int = 5, json_path: str = "models/feature_importance.json") -> List[Tuple[str, float]]:
    """Retrieve top N contributing factors ordered by importance.
    
    Args:
        top_n: Number of top features to return.
        json_path: Path to feature importance JSON.
        
    Returns:
        List of (feature_name, importance_score) tuples.
    """
    importances = load_saved_feature_importance(json_path)
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]


def plot_feature_importance_figure(
    top_n: int = 10,
    json_path: str = "models/feature_importance.json"
) -> plt.Figure:
    """Generate a clean matplotlib horizontal bar chart for feature importance.
    
    Args:
        top_n: Top N features to plot.
        json_path: Path to feature importance JSON.
        
    Returns:
        matplotlib.figure.Figure: Generated bar chart figure.
    """
    importances = load_saved_feature_importance(json_path)
    # Sort ascending for horizontal bar chart (top at the top)
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=False)[-top_n:]
    
    features = [item[0].replace('_', ' ') for item in sorted_items]
    scores = [item[1] * 100 for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(features, scores, color="#2563EB", edgecolor="#1D4ED8", height=0.65)
    
    ax.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="medium")
    ax.set_title(f"Top {len(features)} Factors Contributing to ML Prediction", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                va='center', ha='left', fontsize=9, color="#1E293B", fontweight="medium")
                
    ax.set_xlim(0, max(scores) * 1.18 if scores else 100)
    plt.tight_layout()
    return fig
