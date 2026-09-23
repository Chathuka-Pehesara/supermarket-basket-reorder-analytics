# Experimental Notebooks

This directory contains Jupyter notebooks structured sequentially according to the project's analytical workflow.

---

## Notebook Roadmap & Individual Ownership

| Notebook | Primary Owner | Focus & Methodologies | Key Deliverables |
| :--- | :--- | :--- | :--- |
| `01_eda_preprocessing.ipynb` | **Pehesara W.A.C** | Data loading, schema validation, handling nulls/duplicates, sampling, basket size distributions, category frequencies. | Cleaned datasets, exploratory charts, summary statistics. |
| `02_market_basket_analysis.ipynb` | **Obesekara S.O.K.D** | Transaction matrix transformation, FP-Growth baseline execution, Apriori & ECLAT benchmarking. | Frequent itemsets, association rules ranked by Support, Confidence, Lift, and Conviction. |
| `03_reorder_prediction.ipynb` | **Peiris W.S.V** | Feature engineering from `days_since_prior_order`, training Logistic Regression, Decision Tree, Random Forest models. | Reorder probability models, ROC-AUC comparisons, confusion matrices, feature importance plots. |
| `04_stakeholder_validation.ipynb` | **Weerasooriya W.H.M.S.P**| Cross-checking findings against Sri Lankan POS reference data and stakeholder in-store insights. | Actionable shelf layout guide, product bundle recommendations, restock prioritization rules. |

---

## Notebook Conventions
1. **Import Reusable Logic**: Place complex functions in `src/` rather than defining massive script blocks in notebooks.
2. **Top-to-Bottom Execution**: Ensure the notebook can be executed cleanly from cell 1 without errors.
3. **Outputs**: Clear unnecessary debug dumps before committing to git.
4. **Reproducibility**: Always use `random_state=42` where sampling or modeling occurs.
