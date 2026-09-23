# Source Package (`src`)

The `src` package provides modular, reusable Python code for the Supermarket Retail Analytics project. Placing core logic here keeps notebooks clean, readable, and focused on visual storytelling and interpretation.

---

## Module Overview

```text
src/
├── __init__.py               # Package initializer
├── data_loader.py            # Ingestion, validation, sampling, and missing value handlers
├── basket_transformer.py     # Pivots transactional lines into one-hot or list-based baskets
├── association_engine.py     # Wrappers for FP-Growth, Apriori, ECLAT, and metric filtering
├── reorder_classifier.py     # Feature engineering, model pipelines, and evaluation metrics
└── utils.py                  # Plotting utilities, path resolution, and reporting helpers
```

---

## Usage Example

```python
# In your notebook (e.g. notebooks/02_market_basket_analysis.ipynb):
from src.data_loader import load_processed_data
from src.basket_transformer import create_basket_matrix
from src.association_engine import run_fp_growth, filter_association_rules

# 1. Load data
df = load_processed_data("cleaned_transactions.parquet")

# 2. Build basket matrix
basket_matrix = create_basket_matrix(df, order_col="order_id", item_col="product_name")

# 3. Mine association rules
frequent_itemsets = run_fp_growth(basket_matrix, min_support=0.01)
rules = filter_association_rules(frequent_itemsets, min_confidence=0.2, min_lift=1.2)
```
