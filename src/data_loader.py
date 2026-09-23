"""
Module: data_loader.py
Author: Pehesara W.A.C (IT24103684) / Group 2026-AI-15
Purpose: Data ingestion, schema validation, missing value handling, and sampling.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# Dynamic root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_dataset(
    filename: str,
    nrows: Optional[int] = None,
    use_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    Safely load a raw CSV or Parquet dataset from data/raw/ with optional row limit.
    """
    filepath = DATA_RAW / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {filepath}")

    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath, columns=use_cols)
    return pd.read_csv(filepath, nrows=nrows, usecols=use_cols)


def clean_ecommerce_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names, handle missing values (e.g. days_since_prior_order for first order),
    and optimize memory usage by downcasting types.
    """
    cleaned = df.copy()

    # Fill NaN for first-time orders (days_since_prior_order is NaN for initial user orders)
    if "days_since_prior_order" in cleaned.columns:
        cleaned["days_since_prior_order"] = cleaned["days_since_prior_order"].fillna(-1)

    # Memory optimization: categorical conversions
    cat_candidates = ["department", "aisle", "product_name"]
    for col in cat_candidates:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("category")

    return cleaned


def create_stratified_sample(
    df: pd.DataFrame,
    fraction: float = 0.1,
    group_col: str = "department",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a representative stratified sample to manage RAM during exploratory modeling.
    """
    if group_col not in df.columns:
        return df.sample(frac=fraction, random_state=random_state)

    return (
        df.groupby(group_col, group_keys=False, observed=False)
        .apply(lambda x: x.sample(frac=fraction, random_state=random_state))
        .reset_index(drop=True)
    )
