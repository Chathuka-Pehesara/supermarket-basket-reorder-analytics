"""
Module: utils.py
Author: Group 2026-AI-15
Purpose: Common plotting functions, path management, and formatting helpers.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def setup_plot_style():
    """Apply consistent styling across all project figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 150


def save_figure(fig: plt.Figure, filename: str):
    """Save a generated figure into the reports directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {out_path}")
