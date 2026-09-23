# Project Standards & Development Rules

**Module**: IT3091 – Machine Learning  
**Project**: Supermarket Retail Analytics & Predictive Merchandising  
**Group ID**: 2026-AI-15  

This document defines the mandatory development rules, Git conventions, code quality standards, and error-prevention guidelines for all team members. Following these guidelines ensures smooth collaboration, zero Git conflicts, and high academic evaluation marks.

---

## 1. Git Workflow & Branch Management

### 1.1 Golden Rules
1. **Never push directly to `main` without testing**: The `main` branch must always remain in a working, demonstrable state.
2. **Always pull before starting work**: Run `git pull origin main` to ensure your local branch is synchronized.
3. **Keep commits atomic**: A commit should represent one logical change (e.g., "add data cleaning script" instead of "update project").

### 1.2 Branch Naming Conventions
Always branch from `main`. Use the following naming structure:
`feature/<member_firstname>-<short_task_description>`

* Examples:
  * `feature/pehesara-data-cleaning`
  * `feature/obesekara-fpgrowth-mining`
  * `feature/peiris-reorder-classifier`
  * `feature/weerasooriya-metrics-evaluation`
  * `fix/pehesara-null-values-bug`
  * `docs/weerasooriya-consent-letter`

### 1.3 Pre-Push Checklist
Before running `git push`, complete this 4-point verification:
* [ ] Ran `git status` to verify no unintended or temporary files are staged.
* [ ] Verified that raw datasets (`data/raw/`), large models (`models/*.pkl`), or cache files (`.ipynb_checkpoints/`) are **NOT** staged.
* [ ] Notebooks and Python scripts execute cleanly without fatal runtime errors.
* [ ] Commit messages conform to Conventional Commit standards (see Section 2).

---

## 2. Git Commit Standards (Conventional Commits)

Commit messages must be concise, descriptive, and follow the format:

```text
<type>(<scope>): <short description in imperative mood>

[optional body explaining WHY the change was made]
```

### 2.1 Allowed Types
| Type | Purpose | Example |
| :--- | :--- | :--- |
| `feat` | Adding a new feature or analytical model | `feat(association): implement FP-Growth rule generator` |
| `fix` | Fixing a bug, data leak, or calculation error | `fix(eda): handle missing days_since_prior_order values` |
| `docs` | Documentation, markdown, plans, or academic records | `docs(plan): update stakeholder consultation notes` |
| `refactor` | Code restructuring without changing functionality | `refactor(data): extract basket pivoting into src/basket_transformer.py` |
| `perf` | Performance improvement (e.g., memory optimization) | `perf(data): downcast integer columns to reduce RAM usage` |
| `test` | Adding or updating unit tests/validation scripts | `test(classifier): add cross-validation check for random forest` |
| `chore` | Dependency updates, `.gitignore`, directory maintenance | `chore(deps): add mlxtend to requirements.txt` |

### 2.2 Good vs. Bad Commit Examples
* ❌ **Bad**: `git commit -m "changes"`
* ❌ **Bad**: `git commit -m "fix code and added stuff"`
* ❌ **Bad**: `git commit -m "update final.ipynb"`
* ✅ **Good**: `feat(data): implement stratified sampling utility for 2M row dataset`
* ✅ **Good**: `fix(metrics): resolve division by zero in conviction calculation`
* ✅ **Good**: `docs(readme): add environment setup and ipykernel instructions`

---

## 3. Data Safety, Privacy & Git Hygiene

### 3.1 Dataset Size & Git Protection
* **NEVER commit large files (> 25MB) to GitHub**:
  * Raw datasets (`data/raw/`) and full processed tables (`data/processed/`) are ignored by `.gitignore`.
  * If small sample data is needed for shared testing, save an anonymized sample under `data/samples/` (< 5MB).
* **Verify before adding**:
  ```bash
  # Check file size of staged files
  git diff --cached --stat
  ```

### 3.2 Data Privacy & Ethics
* **Zero PII**: No customer names, phone numbers, email addresses, or physical addresses must ever enter the repository.
* All customer identities must remain pseudo-anonymized integers or hashes (`user_id`).
* Any local stakeholder feedback recorded in `docs/` must have formal consent.

### 3.3 Transparent Data Provenance
* When presenting charts or metrics, clearly state data source:
  * Kaggle dataset findings: labeled as *Proxy E-Commerce Dataset*.
  * Sri Lankan POS findings: labeled as *Local POS Reference Dataset*.
  * Stakeholder notes: labeled as *In-Store Consultation*.

---

## 4. Jupyter Notebook Hygiene & Error Prevention

Jupyter Notebooks are prone to merge conflicts and hidden state errors. All team members must adhere to these rules:

1. **Sequential Execution Order**:
   * Cells must execute cleanly from top to bottom (Cell [1], [2], [3]...).
   * Never leave notebooks in an out-of-order execution state.
   * **Rule**: Before pushing, click `Kernel -> Restart Kernel and Run All Cells` to confirm zero exceptions.

2. **Clear Bulky Outputs Before Pushing**:
   * If a cell prints thousands of lines or large interactive tables, clear the output or limit stdout (`df.head(10)` instead of `df`).
   * Large outputs cause git bloat and merge conflicts.

3. **No Heavy Business Logic in Notebooks**:
   * Notebooks are for **exploration, storytelling, and visualization**.
   * Reusable functions (data loaders, metric calculators, data transformations) **MUST** be placed in Python files under `src/` and imported into notebooks:
     ```python
     # Good Practice in Notebooks:
     from src.data_loader import load_clean_data
     from src.association_engine import run_fp_growth
     ```

---

## 5. Python Code Quality & Reproducibility Standards

### 5.1 Deterministic & Reproducible Results
Machine learning models must yield identical results across all team members' machines:
* **Set Random Seed**: Always supply `random_state=42` in scikit-learn models, train/test splits, and sampling methods:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  rf = RandomForestClassifier(n_estimators=100, random_state=42)
  ```

### 5.2 Dynamic Path Resolution (Never Hardcode Absolute Paths)
* ❌ **Bad**:
  ```python
  df = pd.read_csv("C:/Users/Pehesara/Downloads/ecommerce.csv") # Will fail on teammate's laptop!
  ```
* ✅ **Good**:
  ```python
  from pathlib import Path

  # Project root is 1 level above src/
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_2023.csv"
  df = pd.read_csv(DATA_PATH)
  ```

### 5.3 Memory Efficiency Guidelines
Working with 2,000,000 transaction rows can easily exhaust RAM if not managed:
1. **Optimize Dtypes**:
   * Convert string categorical columns with few unique values to `'category'`.
   * Downcast `int64` to `int32` or `int16` where values permit.
2. **Use Parquet**:
   * Save intermediate cleaned tables as `.parquet` instead of `.csv` for 5x–10x compression and fast loading.
3. **Garbage Collection**:
   * Delete large temporary tables when no longer needed:
     ```python
     import gc
     del raw_df
     gc.collect()
     ```

---

## 6. Resolving Merge Conflicts & Collaboration Protocol

1. **Work in Different Notebooks**:
   * Each member owns their dedicated notebook:
     * `01_eda_preprocessing.ipynb` (Pehesara)
     * `02_market_basket_analysis.ipynb` (Obesekara)
     * `03_reorder_prediction.ipynb` (Peiris)
     * `04_stakeholder_validation.ipynb` (Weerasooriya)
   * Avoid editing someone else's notebook simultaneously.

2. **In Case of Conflict**:
   * Never blindly overwrite files.
   * Pull changes using rebase: `git pull --rebase origin main`.
   * For conflicting `.ipynb` files, resolve via VS Code diff tool or save a copy before syncing.

3. **Code Reviews & Quality Checks**:
   * Before merging a major feature branch into `main`, ask another team member to review the logic and ensure it runs on their local environment.
