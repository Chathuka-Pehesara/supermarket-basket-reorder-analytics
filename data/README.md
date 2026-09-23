# Data Management & Storage

This directory manages all datasets used across the project lifecycle. 

> **Important**: Datasets stored in `data/raw/` and `data/processed/` are excluded from Git version control via `.gitignore` to prevent repository bloat and respect data licensing.

---

## Directory Organization

```text
data/
├── raw/         <- Immutable raw data sources (Kaggle dataset, Sri Lankan POS logs)
├── processed/   <- Cleaned, transformed data (e.g. Parquet tables, one-hot basket matrices)
└── samples/     <- Lightweight mock/sample datasets (< 5MB) for quick testing & CI/CD
```

---

## Dataset Provenance & Setup

### 1. Primary Analytical Dataset (Kaggle E-Commerce 2023)
* **Download Source**: [Kaggle E-Commerce Dataset for Predictive Marketing (2023)](https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023)
* **Destination**: Place downloaded CSV file(s) into `data/raw/`.
* **Nature**: Disclosed analytical proxy used for market basket analysis and reorder velocity modeling.

### 2. Secondary Reference Dataset (Sri Lankan POS Dataset)
* **Scale**: ~551,639 line items from 5 physical branch locations (2024–2026).
* **Destination**: Place in `data/raw/sri_lankan_pos/`.
* **Purpose**: Validation and sanity-checking of FMCG co-purchases (tea, dairy, biscuits) within the local retail context.

### 3. Processed Datasets
* Transformed tables and basket matrices should be exported in **Parquet format** (`.parquet`) into `data/processed/` using `src/basket_transformer.py` to optimize read speeds and reduce RAM overhead.
