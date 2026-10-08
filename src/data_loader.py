"""
Module: data_loader.py
Author: Pehesara W.A.C (IT24103684) / Group 2026-AI-15
Purpose: Production-grade data ingestion, structural validation, memory optimization,
         feature enrichment, and persistent storage routines.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

# Global Path Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLES = PROJECT_ROOT / "data" / "samples"

DEFAULT_RAW_FILE = "ECommerce_consumer_behaviour.csv"


def load_raw_dataset(
    filename: str = DEFAULT_RAW_FILE,
    nrows: Optional[int] = None,
    usecols: Optional[list] = None
) -> pd.DataFrame:
    """
    Safely load the raw dataset from `data/raw/` with defensive path resolution.

    Parameters:
        filename (str): Name of the raw CSV or Parquet file.
        nrows (int, optional): Number of rows to read (useful for rapid testing).
        usecols (list, optional): Subset of columns to read.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.
    """
    filepath = DATA_RAW / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected location: {filepath}.\n"
            f"Please ensure {filename} exists inside {DATA_RAW}."
        )

    print(f"[Loading Dataset] -> {filepath.name} (Source: {DATA_RAW})")
    if filepath.suffix == ".parquet":
        return pd.read_parquet(filepath, columns=usecols)
    return pd.read_csv(filepath, nrows=nrows, usecols=usecols)


def audit_dataset_health(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs a comprehensive diagnostic audit of the dataset:
    dimensions, memory footprint, null counts, duplicate records, and cardinalities.

    Parameters:
        df (pd.DataFrame): Input DataFrame to audit.

    Returns:
        Dict[str, Any]: Diagnostic health dictionary.
    """
    total_rows, total_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    null_counts = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())

    unique_counts = {
        col: int(df[col].nunique()) for col in df.columns
    }

    audit_report = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "memory_mb": round(memory_mb, 2),
        "duplicate_rows": duplicate_rows,
        "null_counts": null_counts,
        "cardinalities": unique_counts
    }
    return audit_report


def optimize_dtypes_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cleans structural nulls and downcasts memory types:
    - Identifies missing `days_since_prior_order` as customer's first order (structural null).
    - Creates boolean indicator `is_first_order`.
    - Imputes `days_since_prior_order` NaN with -1 (meaning initial baseline order).
    - Downcasts integers to `uint8`, `uint16`, `uint32`, and strings to `category`.

    Parameters:
        df (pd.DataFrame): Raw transaction DataFrame.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: (Optimized DataFrame, Memory statistics dict).
    """
    cleaned = df.copy()
    initial_mem_mb = cleaned.memory_usage(deep=True).sum() / (1024 ** 2)

    # 1. Structural null handling for days_since_prior_order
    if "days_since_prior_order" in cleaned.columns:
        cleaned["is_first_order"] = cleaned["days_since_prior_order"].isna().astype("bool")
        cleaned["days_since_prior_order"] = cleaned["days_since_prior_order"].fillna(-1).astype("int8")

    # 2. Downcasting integer identifiers and orders
    dtype_mapping = {
        "order_id": "uint32",
        "user_id": "uint32",
        "order_number": "uint8",
        "order_dow": "uint8",
        "order_hour_of_day": "uint8",
        "product_id": "uint16",
        "add_to_cart_order": "uint8",
        "reordered": "uint8",
        "department_id": "uint8"
    }

    for col, dtype in dtype_mapping.items():
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(dtype)

    # 3. Categorical encoding for high-repetition strings
    for col in ["department", "product_name"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("category")

    final_mem_mb = cleaned.memory_usage(deep=True).sum() / (1024 ** 2)
    savings_pct = (1 - (final_mem_mb / initial_mem_mb)) * 100

    mem_stats = {
        "initial_mb": round(initial_mem_mb, 2),
        "optimized_mb": round(final_mem_mb, 2),
        "savings_pct": round(savings_pct, 1)
    }

    print(
        f"[Memory Optimization] {initial_mem_mb:.1f} MB -> {final_mem_mb:.1f} MB "
        f"({savings_pct:.1f}% reduction)"
    )
    return cleaned, mem_stats


def clean_ecommerce_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names, handle structural missing values, and downcast dtypes.
    Wrapper returning only the cleaned DataFrame.
    """
    cleaned, _ = optimize_dtypes_and_clean(df)
    return cleaned


def engineer_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives business-critical features for exploratory analysis:
    - `basket_size`: Total count of items in each order.
    - `order_day_name`: Human-readable day of week (0=Sunday to 6=Saturday).
    - `order_time_slot`: Segmented time of day (Morning, Afternoon, Evening, Night).

    Parameters:
        df (pd.DataFrame): Cleaned transactions DataFrame.

    Returns:
        pd.DataFrame: Enriched DataFrame.
    """
    enriched = df.copy()

    # Basket size per order
    if "order_id" in enriched.columns:
        enriched["basket_size"] = enriched.groupby("order_id", observed=False)["product_id"].transform("count").astype("uint8")

    # Day of week mapping (Standard Instacart/Kaggle convention: 0=Sunday, 1=Monday, ... 6=Saturday)
    dow_map = {
        0: "Sunday",
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday"
    }
    if "order_dow" in enriched.columns:
        enriched["order_day_name"] = enriched["order_dow"].map(dow_map).astype("category")

    # Time slot segmentation
    def categorize_hour(hour: int) -> str:
        if 6 <= hour < 12:
            return "Morning (06-11)"
        elif 12 <= hour < 17:
            return "Afternoon (12-16)"
        elif 17 <= hour < 22:
            return "Evening (17-21)"
        else:
            return "Night (22-05)"

    if "order_hour_of_day" in enriched.columns:
        enriched["order_time_slot"] = enriched["order_hour_of_day"].apply(categorize_hour).astype("category")

    return enriched


def save_processed_dataset(
    df: pd.DataFrame,
    filename: str = "cleaned_transactions.parquet"
) -> Path:
    """
    Persists the cleaned, optimized dataset to `data/processed/` using Parquet compression.

    Parameters:
        df (pd.DataFrame): Cleaned and enriched dataset.
        filename (str): Target filename (.parquet recommended).

    Returns:
        Path: Absolute path to the saved file.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / filename

    if filename.endswith(".parquet"):
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    elif filename.endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Filename must end with .parquet or .csv")

    file_size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"[Persisted Data] -> {out_path} ({file_size_mb:.2f} MB)")
    return out_path


def create_stratified_sample(
    df: pd.DataFrame,
    fraction: float = 0.1,
    group_col: str = "department",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a representative stratified sample across product departments
    to support rapid prototyping and test suite runs.
    """
    if group_col not in df.columns:
        return df.sample(frac=fraction, random_state=random_state)

    return (
        df.groupby(group_col, group_keys=False, observed=False)
        .apply(lambda x: x.sample(frac=fraction, random_state=random_state))
        .reset_index(drop=True)
    )
