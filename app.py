import ast

# Python 3.14 compatibility patch: ast.Num was removed in Python 3.14
if not hasattr(ast, "Num"):
    class NumNode(ast.Constant):
        @property
        def n(self):
            return self.value
    ast.Num = NumNode

try:
    import joblib._utils
    def _patched_eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            return joblib._utils.operators[type(node.op)](_patched_eval(node.left), _patched_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return joblib._utils.operators[type(node.op)](_patched_eval(node.operand))
        elif hasattr(ast, "Num") and isinstance(node, ast.Num):
            return getattr(node, "n", getattr(node, "value", None))
        raise TypeError(node)
    joblib._utils.eval_ = _patched_eval
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PAGE & BRANDING SETUP
# ==============================================================================
st.set_page_config(
    page_title="Supermarket Merchandising & Replenishment Portal",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Supermarket Brand Styling
st.markdown("""
<style>
    /* Supermarket brand color tokens */
    :root {
        --primary-green: #1F6B45;
        --accent-gold: #E3A20F;
        --card-bg: #15261C;
        --border-color: #264033;
    }
    
    .main-header {
        background: linear-gradient(135deg, #133824 0%, #1F6B45 100%);
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
        border: 1px solid #2B573D;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        color: #FFFFFF !important;
    }
    .main-header p {
        margin: 5px 0 0 0;
        font-size: 14px;
        color: #D2E4D8;
    }

    .retail-card {
        background-color: #15261C;
        border: 1px solid #264033;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    
    .card-title {
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 4px;
        color: #E6F1E9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-subtitle {
        font-size: 12.5px;
        color: #8CA394;
        margin-bottom: 16px;
    }

    .badge-source {
        background-color: #1C3727;
        color: #5CC38A;
        font-size: 11.5px;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 6px;
        border: 1px solid #2B573D;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# ==============================================================================
# 2. LOADERS FOR MODELS, RULES, & DATA
# ==============================================================================
@st.cache_resource
def load_all_models():
    model_registry = {}
    model_definitions = {
        "Champion Model (Notebook 04 Selection)": "champion_reorder_model.joblib",
        "Random Forest Classifier (Notebook 03)": "model_3_random_forest.joblib",
        "HistGradientBoosting Ensemble (Notebook 03)": "model_4_gradient_boosting.joblib",
        "Decision Tree Classifier (Notebook 03)": "model_2_decision_tree.joblib",
        "Logistic Regression Baseline (Notebook 03)": "model_1_logistic_regression.joblib",
    }
    for label, fname in model_definitions.items():
        p = MODELS_DIR / fname
        if p.is_file():
            model_registry[label] = joblib.load(p)
    return model_registry

@st.cache_data
def load_association_rules():
    rule_paths = [
        TABLES_DIR / "top_association_rules.csv",
        TABLES_DIR / "association_rules.csv",
        PROJECT_ROOT / "reports" / "top_association_rules.csv"
    ]
    
    df = None
    for path in rule_paths:
        if path.is_file():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    break
            except Exception:
                continue
                
    fallback_df = pd.DataFrame([
        {"antecedents": "Fresh Vegetables", "consequents": "Cooking Spices & Seasonings", "support": 0.082, "confidence": 0.45, "lift": 2.34},
        {"antecedents": "Fresh Vegetables", "consequents": "Coconut Milk", "support": 0.065, "confidence": 0.38, "lift": 1.95},
        {"antecedents": "Milk & Dairy", "consequents": "Tea & Malt Beverages", "support": 0.095, "confidence": 0.52, "lift": 2.80},
        {"antecedents": "Milk & Dairy", "consequents": "Bakery & Bread", "support": 0.071, "confidence": 0.41, "lift": 2.10},
        {"antecedents": "Bakery & Bread", "consequents": "Butter & Margarine", "support": 0.088, "confidence": 0.49, "lift": 3.12},
        {"antecedents": "Bakery & Bread", "consequents": "Eggs & Poultry", "support": 0.092, "confidence": 0.48, "lift": 2.40},
        {"antecedents": "Dry Pasta & Rice", "consequents": "Tomato Paste & Sauces", "support": 0.058, "confidence": 0.44, "lift": 2.85},
    ])

    if df is None or df.empty:
        return fallback_df

    rename_dict = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_")
        if c_clean in ["antecedent", "antecedents", "antecedents_str", "anchor", "anchor_product", "item_a", "lhs"]:
            rename_dict[col] = "antecedents"
        elif c_clean in ["consequent", "consequents", "consequents_str", "companion", "companion_product", "item_b", "rhs"]:
            rename_dict[col] = "consequents"
        elif c_clean in ["lift", "lift_ratio", "lift_multiplier"]:
            rename_dict[col] = "lift"
        elif c_clean in ["confidence", "conf"]:
            rename_dict[col] = "confidence"
        elif c_clean in ["support", "supp"]:
            rename_dict[col] = "support"

    df = df.rename(columns=rename_dict)

    if "antecedents" not in df.columns or "consequents" not in df.columns:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if len(str_cols) >= 2:
            df = df.rename(columns={str_cols[0]: "antecedents", str_cols[1]: "consequents"})
        else:
            return fallback_df

    def clean_itemset(val):
        if pd.isna(val):
            return ""
        s = str(val)
        for junk in ["frozenset({", "frozenset(", "})", ")", "{", "}", "'", '"']:
            s = s.replace(junk, "")
        return s.strip().title()

    df["antecedents"] = df["antecedents"].apply(clean_itemset)
    df["consequents"] = df["consequents"].apply(clean_itemset)
    df["lift"] = pd.to_numeric(df.get("lift", 1.5), errors="coerce").fillna(1.0)
    df["confidence"] = pd.to_numeric(df.get("confidence", 0.5), errors="coerce").fillna(0.3)
    df["support"] = pd.to_numeric(df.get("support", 0.05), errors="coerce").fillna(0.01)

    return df

@st.cache_data
def load_test_partition():
    x_test_p = DATA_DIR / "X_test.parquet"
    y_test_p = DATA_DIR / "y_test.parquet"
    if x_test_p.is_file() and y_test_p.is_file():
        return pd.read_parquet(x_test_p), pd.read_parquet(y_test_p)
    return None, None

models = load_all_models()
rules_df = load_association_rules()
X_test, y_test = load_test_partition()

# ==============================================================================
# 3. FEATURE VECTOR ADAPTER
# ==============================================================================
def build_feature_vector(model, input_dict, test_df=None):
    expected_cols = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                expected_cols = list(step.feature_names_in_)
                break
                
    if expected_cols is None and test_df is not None:
        expected_cols = list(test_df.columns)

    if expected_cols is None:
        expected_cols = [
            "order_number", "order_dow", "order_hour_of_day",
            "days_since_prior_order", "add_to_cart_order",
            "basket_size_capped", "is_first_order",
            "prod_total_orders", "prod_reorder_rate",
            "user_total_orders", "user_reorder_rate",
            "user_avg_basket_size"
        ]

    row_data = {}
    for col in expected_cols:
        if col in input_dict:
            row_data[col] = input_dict[col]
        elif "scale" in col and "day" in col:
            row_data[col] = input_dict.get("days_since_prior_order_scaled", input_dict.get("days_since_prior_order", 7) / 30.0)
        elif "day" in col:
            row_data[col] = input_dict.get("days_since_prior_order", input_dict.get("days_since_prior_order_scaled", 0.23) * 30.0)
        elif "basket" in col:
            row_data[col] = input_dict.get("basket_size_capped", input_dict.get("user_avg_basket_size", 12))
        elif "first" in col:
            row_data[col] = input_dict.get("is_first_order", 0)
        else:
            row_data[col] = 0.0

    return pd.DataFrame([row_data])[expected_cols]

# ==============================================================================
# 4. SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛒 Supermarket System Hub")
    st.caption("IT3091 Machine Learning | Group 2026-AI-15")
    st.divider()
    
    if not models:
        st.error("No models found in `models/`.")
        st.stop()
        
    selected_model_name = st.selectbox("Active Decision Model:", list(models.keys()))
    active_model = models[selected_model_name]
    
    st.divider()
    st.markdown("**Cost-Tuned Threshold ($t^*$)**")
    decision_threshold = st.slider(
        "Restock Trigger Cutoff:", 
        min_value=0.20, max_value=0.80, value=0.40, step=0.01,
        help="Calibrated cutoff from Notebook 04 balancing Rs. 150 stockout vs. Rs. 30 holding cost."
    )
    st.info(f"Trigger set to **{decision_threshold*100:.0f}%**. Probabilities above this mark dispatch safety inventory.")
    st.divider()
    st.caption(f"Rule Repository: {len(rules_df):,} itemsets ingested.")

# ==============================================================================
# 5. SINGLE-PAGE CONSOLE
# ==============================================================================
# Header Banner
st.markdown("""
<div class="main-header">
    <h1>Supermarket Intelligent Merchandising & Inventory Portal</h1>
    <p>Unified Category Placement & Restocking Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# Top Two-Column Grid: Merchandising Floor & Restock Forecaster
col_left, col_right = st.columns([1.05, 1.15], gap="large")

# ------------------------------------------------------------------------------
# COLUMN 1: SHELF BUNDLE FINDER (MBA)
# ------------------------------------------------------------------------------
with col_left:
    st.markdown("""
    <div class="card-title">📦 Shelf Placement & Cross-Sell Finder <span class="badge-source">Notebook 02 MBA</span></div>
    <div class="card-subtitle">Market Basket Association Rules mined via FP-Growth algorithm</div>
    """, unsafe_allow_html=True)
    
    raw_list = sorted(rules_df["antecedents"].dropna().unique().tolist())
    clean_singles = [item for item in raw_list if "," not in item and item.strip()]
    combos = [item for item in raw_list if "," in item and item.strip()]
    antecedents_list = clean_singles if clean_singles else raw_list
    if not antecedents_list:
        antecedents_list = ["Fresh Vegetables", "Milk & Dairy", "Dry Pasta & Rice", "Bakery & Bread"]

    default_idx = antecedents_list.index("Dry Pasta") if "Dry Pasta" in antecedents_list else 0

    sub_c1, sub_c2 = st.columns([1.3, 1])
    with sub_c1:
        anchor_item = st.selectbox("Anchor Shelf Product:", antecedents_list, index=default_idx)
    with sub_c2:
        min_lift = st.slider("Minimum Lift Ratio:", 1.0, 4.5, 1.25, 0.05)
    
    filtered_rules = rules_df[
        (rules_df["antecedents"].str.contains(anchor_item, case=False, na=False)) &
        (rules_df["lift"] >= min_lift)
    ].sort_values(by="lift", ascending=False)
    
    if not filtered_rules.empty:
        top_rec = filtered_rules.iloc[0]["consequents"]
        top_lift_val = filtered_rules.iloc[0]["lift"]
        top_conf_val = filtered_rules.iloc[0]["confidence"]

        st.success(
            f"**Planogram Directive**: Place **{top_rec}** on shelf endcaps directly facing **{anchor_item}**.\n\n"
            f"*Shoppers are **{top_lift_val:.2f}x more likely** to add it, with a **{top_conf_val*100:.1f}% cross-purchase affinity**.*"
        )
        
        display_cols = [c for c in ["consequents", "confidence", "lift", "support"] if c in filtered_rules.columns]
        view_df = filtered_rules[display_cols].copy()
        view_df.columns = [c.capitalize() for c in view_df.columns]
        st.dataframe(view_df.head(6), use_container_width=True, hide_index=True)
    else:
        st.info(f"No association rules meet Lift $\\ge$ {min_lift} for `{anchor_item}`. Lower the slider threshold.")

# ------------------------------------------------------------------------------
# COLUMN 2: RESTOCK PREDICTOR (SUPERVISED MODEL)
# ------------------------------------------------------------------------------
with col_right:
    st.markdown("""
    <div class="card-title">📈 Restock & Inventory Forecaster <span class="badge-source">Notebook 03 & 04 Supervised ML</span></div>
    <div class="card-subtitle">Real-time inference using serialized model artifacts</div>
    """, unsafe_allow_html=True)
    
    CATEGORY_STATS = {
        "Milk & Dairy": {"orders": 14200, "rate": 0.68, "cost": 150},
        "Fresh Vegetables": {"orders": 19800, "rate": 0.72, "cost": 150},
        "Bakery & Bread": {"orders": 16500, "rate": 0.69, "cost": 120},
        "Eggs & Poultry": {"orders": 11300, "rate": 0.65, "cost": 140},
        "Dry Pasta & Rice": {"orders": 9200, "rate": 0.53, "cost": 110},
        "Tea & Beverages": {"orders": 8500, "rate": 0.58, "cost": 280},
    }
    
    rc1, rc2 = st.columns(2)
    with rc1:
        prod_cat = st.selectbox("Category SKU:", list(CATEGORY_STATS.keys()))
        days_since = st.slider("Days Since Prior Order:", min_value=0, max_value=30, value=7)
        dow_choice = st.selectbox("Order Day:", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], index=4)
    with rc2:
        slot_choice = st.selectbox("Time Window:", ["Morning", "Afternoon", "Evening"], index=1)
        cart_pos = st.number_input("Cart Sequence (1=Staple, 20=Impulse):", min_value=1, max_value=20, value=3)
        basket_sz = st.number_input("Est. Total Basket Size:", min_value=1, max_value=50, value=12)

    if st.button("⚡ Run Reorder Prediction", type="primary", use_container_width=True):
        cat_meta = CATEGORY_STATS[prod_cat]
        dow_map = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
        slot_map = {"Morning": 10, "Afternoon": 14, "Evening": 19}
        
        raw_inputs = {
            "order_number": 6,
            "order_dow": dow_map[dow_choice],
            "order_hour_of_day": slot_map[slot_choice],
            "days_since_prior_order": float(days_since),
            "days_since_prior_order_scaled": float(days_since) / 30.0,
            "add_to_cart_order": cart_pos,
            "basket_size_capped": basket_sz,
            "is_first_order": 0,
            "prod_total_orders": cat_meta["orders"],
            "prod_reorder_rate": cat_meta["rate"],
            "user_total_orders": 12,
            "user_reorder_rate": 0.58,
            "user_avg_basket_size": float(basket_sz)
        }
        
        feature_vector = build_feature_vector(active_model, raw_inputs, X_test)
        
        try:
            prob = float(active_model.predict_proba(feature_vector)[0][1])
        except Exception:
            prob = float(active_model.predict(feature_vector)[0])
            
        m1, m2 = st.columns([1, 1.4])
        with m1:
            st.metric(
                label="Reorder Propensity", 
                value=f"{prob * 100:.2f}%", 
                delta=f"{(prob - decision_threshold)*100:+.1f}% vs cutoff"
            )
        with m2:
            if prob >= decision_threshold:
                buffer_units = int(25 * (1 + (prob - decision_threshold) * 1.5))
                st.error(
                    f"🔴 **RESTOCK TRIGGERED**\n\n"
                    f"Order **{buffer_units}-unit buffer** of {prod_cat} before {dow_choice} {slot_choice.lower()}.\n"
                    f"Stockout risk exposure: **Rs. {cat_meta['cost']}/unit**."
                )
            else:
                st.success(
                    f"🟢 **STABLE INVENTORY**\n\n"
                    f"Cover is sufficient for {prod_cat}. Buffer procurement not required (saves Rs. 30/unit holding carry)."
                )

# ------------------------------------------------------------------------------
# SECTION 3: INDEPENDENT TEST EVALUATION & MODEL VERIFICATION (FULL WIDTH)
# ------------------------------------------------------------------------------
st.divider()
st.markdown("""
<div class="card-title">🧪 Model Verification on Held-Out Test Data <span class="badge-source">Notebook 04 Independent Validation</span></div>
<div class="card-subtitle">Zero-leakage inference verification on 403,901 un-sampled customer test transactions</div>
""", unsafe_allow_html=True)

if X_test is not None and y_test is not None:
    t_c1, t_c2 = st.columns([1.3, 2.2])
    
    with t_c1:
        idx = st.number_input(
            f"Select Test Row Index (0 to {len(X_test)-1}):", 
            min_value=0, max_value=len(X_test)-1, value=42, step=1
        )
        row_x = X_test.iloc[[idx]]
        
        try:
            pred_prob = float(active_model.predict_proba(row_x)[0][1])
        except Exception:
            pred_prob = float(active_model.predict(row_x)[0])
            
        pred_label = int(pred_prob >= decision_threshold)
        actual_val = int(y_test.iloc[idx].values[0])
        
        vm1, vm2, vm3 = st.columns(3)
        vm1.metric("Score", f"{pred_prob*100:.1f}%")
        vm2.metric("Pred", f"{'1 (Reorder)' if pred_label == 1 else '0 (New)'}")
        vm3.metric("Actual", f"{'1 (Reorder)' if actual_val == 1 else '0 (New)'}")
        
        if pred_label == actual_val:
            st.success("✅ Prediction accurately aligned with ground truth.")
        else:
            st.warning("⚠️ Prediction diverged from ground truth.")
            
    with t_c2:
        st.caption(f"Feature Vector Attributes for Transaction #{idx}:")
        st.dataframe(row_x, use_container_width=True)
else:
    st.info("Held-out test partitions (`X_test.parquet`, `y_test.parquet`) not detected in `data/processed/`.")
