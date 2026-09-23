"""
Module: association_engine.py
Author: Obesekara S.O.K.D (IT24103866) / Evaluation: Weerasooriya W.H.M.S.P (IT24104192)
Purpose: FP-Growth, Apriori, and ECLAT association mining wrappers with metric filtering.
"""

from typing import Optional
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules


def extract_frequent_itemsets(
    basket_df: pd.DataFrame,
    algorithm: str = "fpgrowth",
    min_support: float = 0.01,
    use_colnames: bool = True
) -> pd.DataFrame:
    """
    Run FP-Growth or Apriori to extract frequent itemsets satisfying minimum support.
    """
    algo = algorithm.lower()
    if algo == "fpgrowth":
        return fpgrowth(basket_df, min_support=min_support, use_colnames=use_colnames)
    elif algo == "apriori":
        return apriori(basket_df, min_support=min_support, use_colnames=use_colnames)
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'fpgrowth' or 'apriori'.")


def generate_pruned_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "lift",
    min_threshold: float = 1.0,
    min_confidence: float = 0.15
) -> pd.DataFrame:
    """
    Generate association rules and compute Support, Confidence, Lift, and Conviction.
    Filters out trivial rules.
    """
    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold
    )

    # Additional filtering on confidence and lift
    pruned = rules[
        (rules["confidence"] >= min_confidence) &
        (rules["lift"] > 1.0)
    ].sort_values(by="lift", ascending=False)

    return pruned
