"""
Module: basket_transformer.py
Author: Obesekara S.O.K.D (IT24103866) / Group 2026-AI-15
Purpose: Transform line-item transaction records into transaction baskets for association mining.
"""

from typing import List, Union
import pandas as pd
from scipy.sparse import csr_matrix


def generate_transaction_lists(
    df: pd.DataFrame,
    order_col: str = "order_id",
    item_col: str = "product_name"
) -> List[List[str]]:
    """
    Group transactions into a list of item lists for algorithms like FP-Growth or ECLAT.
    """
    return df.groupby(order_col)[item_col].apply(list).tolist()


def create_one_hot_basket(
    df: pd.DataFrame,
    order_col: str = "order_id",
    item_col: str = "product_name",
    min_item_frequency: int = 50
) -> pd.DataFrame:
    """
    Pivots transactions into a binary one-hot encoded DataFrame (Order x Product).
    Filters out ultra-rare items to prevent memory explosion.
    """
    # Filter items that appear fewer than min_item_frequency times
    item_counts = df[item_col].value_counts()
    frequent_items = item_counts[item_counts >= min_item_frequency].index
    filtered_df = df[df[item_col].isin(frequent_items)]

    # Cross-tabulate / one-hot encode
    basket = (
        filtered_df.groupby([order_col, item_col], observed=False)[item_col]
        .count()
        .unstack()
        .fillna(0)
    )

    # Convert counts to boolean / binary indicator (0 or 1)
    binary_basket = basket.map(lambda x: True if x > 0 else False)
    return binary_basket
