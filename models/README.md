# Trained Models & Serialized Estimators

This directory stores serialized model artifacts (e.g. `.joblib`, `.pkl`) trained during the supervised reorder classification workflow.

---

## Directory Policy

* **Git Excluded**: Binary model files are excluded from Git via `.gitignore` to avoid large file issues.
* **Naming Convention**:
  * `reorder_logreg_baseline.joblib`
  * `reorder_decision_tree.joblib`
  * `reorder_random_forest.joblib`
* **Metadata Logging**: Whenever a model is saved, its hyperparameter settings and evaluation scores should be logged in `reports/model_benchmarks.csv`.

---

## Example Model Serialization

```python
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

# Save model
joblib.dump(model, MODEL_DIR / "reorder_random_forest.joblib")

# Load model
loaded_model = joblib.load(MODEL_DIR / "reorder_random_forest.joblib")
```
