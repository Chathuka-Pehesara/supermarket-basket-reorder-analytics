"""
Module: reorder_classifier.py
Author: Peiris W.S.V (IT24103792) / Evaluation: Weerasooriya W.H.M.S.P (IT24104192)
Purpose: Supervised model pipelines for product-level reorder frequency and replenishment timing.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def build_product_reorder_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level records to product-level features for reorder classification.
    Predictors:
    - Average days between purchases (days_since_prior_order)
    - Total order frequency
    - Proportion of times product was reordered
    - Department reorder velocity
    """
    product_stats = df.groupby("product_id").agg(
        total_orders=("order_id", "count"),
        reorder_count=("reordered", "sum"),
        avg_reorder_gap=("days_since_prior_order", lambda x: x[x >= 0].mean()),
        department=("department", "first")
    ).reset_index()

    # Calculate target: High reorder item (binary)
    product_stats["reorder_rate"] = product_stats["reorder_count"] / product_stats["total_orders"]
    median_rate = product_stats["reorder_rate"].median()
    product_stats["is_high_reorder"] = (product_stats["reorder_rate"] >= median_rate).astype(int)

    return product_stats


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train baseline Logistic Regression classifier.
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Compute standard classification evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
    }
    return metrics
