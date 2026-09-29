"""General utility functions and formatting helpers."""
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt


def format_percentage(val: float) -> str:
    """Format float into percentage string."""
    return f"{val * 100:.1f}%"


def get_competitiveness_badge_color(competitiveness: str) -> str:
    """Return theme color for competitiveness category."""
    mapping = {
        "High": "#16A34A",    # Green
        "Medium": "#D97706",  # Amber/Orange
        "Low": "#DC2626"      # Red
    }
    return mapping.get(competitiveness, "#2563EB")


def create_distribution_comparison_chart(
    feature_name: str,
    student_value: float,
    benchmark_stat: Dict[str, float]
) -> plt.Figure:
    """Create a simple box/bar chart showing student value relative to cohort range."""
    fig, ax = plt.subplots(figsize=(6, 1.8))
    
    min_val = benchmark_stat["min"]
    max_val = benchmark_stat["max"]
    median_val = benchmark_stat["median"]
    q25 = benchmark_stat["q25"]
    q75 = benchmark_stat["q75"]
    
    # Draw cohort IQR bar
    ax.barh(y=0, width=q75 - q25, left=q25, height=0.35, color="#E2E8F0", edgecolor="#94A3B8", label="Middle 50% Cohort (IQR)")
    # Draw cohort min-max line
    ax.plot([min_val, max_val], [0, 0], color="#64748B", linestyle="-", linewidth=2, zorder=2)
    # Draw median marker
    ax.scatter([median_val], [0], color="#2563EB", s=100, zorder=4, marker="d", label=f"Cohort Median ({median_val:g})")
    # Draw student marker
    ax.scatter([student_value], [0], color="#DC2626", s=120, zorder=5, marker="o", label=f"Student Value ({student_value:g})")
    
    ax.set_yticks([])
    ax.set_xlabel(feature_name.replace('_', ' '), fontsize=10)
    ax.set_xlim(min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig
