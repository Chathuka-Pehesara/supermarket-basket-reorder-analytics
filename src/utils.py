"""
Module: utils.py
Author: Group 2026-AI-15
Purpose: Common plotting functions, styling presets, path management, and figure export utilities.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Global Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


def setup_plot_style():
    """
    Applies consistent, publication-ready corporate visual styling across all figures.
    Uses clean gridlines, readable font sizes, and an accessible modern palette.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": False
    })


def save_figure(fig: plt.Figure, filename: str):
    """
    Saves a generated matplotlib/seaborn figure into `reports/figures/` at 300 DPI.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Exported Figure] -> {out_path.relative_to(PROJECT_ROOT)}")


def export_summary_table(df: pd.DataFrame, filename: str):
    """
    Exports summary statistics or aggregated tables into `reports/tables/`.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / filename
    if filename.endswith(".csv"):
        df.to_csv(out_path, index=True)
    elif filename.endswith(".parquet"):
        df.to_parquet(out_path, index=True)
    print(f"[Exported Table]  -> {out_path.relative_to(PROJECT_ROOT)}")
