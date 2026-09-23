# Supermarket Retail Analytics & Predictive Merchandising

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use%20Only-lightgrey.svg)](#)
[![Track: Industry Explorer](https://img.shields.io/badge/Track-Industry%20Explorer-brightgreen.svg)](#)
[![SLIIT: IT3091](https://img.shields.io/badge/SLIIT-IT3091%20Machine%20Learning-orange.svg)](#)

**Institution**: Sri Lanka Institute of Information Technology (SLIIT)  
**Module**: IT3091 – Machine Learning (Year 3 Semester 1, 2026)  
**Group ID**: `2026-AI-15`  
**Track**: **Industry Explorer Track** (External datasets & direct local stakeholder consultation)

---

## 1. Project Overview

In traditional supermarket retail, particularly in independent, owner-operated shops, inventory decisions, product placement, and restocking rely almost entirely on operator intuition rather than empirical purchase patterns. This leads to missed cross-selling opportunities, sub-optimal shelf layouts, and unexpected stockouts of fast-moving items.

This project delivers an end-to-end evidence-based merchandising strategy using:
1. **Unsupervised Association Rule Mining (Primary Lens)**: Discovering non-obvious product affinity bundles using **FP-Growth**, benchmarked against **Apriori** and **ECLAT** (*Support, Confidence, Lift, Conviction*).
2. **Supervised Reorder & Restocking Classification (Secondary Lens)**: Predicting product-level reorder frequency and replenishment velocity using **Logistic Regression**, benchmarked against **Decision Tree** and **Random Forest**.
3. **Local Stakeholder Validation**: Grounding all findings through direct consultation with an independent supermarket owner and cross-referencing against a secondary Sri Lankan supermarket POS dataset (~551k line items).

---

## 2. Team Members & Implementation Roles

| Student ID | Student Name | Core Implementation Responsibility | Primary Notebook / Component |
| :--- | :--- | :--- | :--- |
| **IT24103684** | Pehesara W.A.C | Data Collection, Ingestion, Cleaning & EDA | `notebooks/01_eda_preprocessing.ipynb` |
| **IT24103866** | Obesekara S.O.K.D | Product Association Mining (FP-Growth, Apriori, ECLAT) | `notebooks/02_market_basket_analysis.ipynb` |
| **IT24103792** | Peiris W.S.V | Reorder Dynamics Feature Engineering & Supervised Modeling | `notebooks/03_reorder_prediction.ipynb` |
| **IT24104192** | Weerasooriya W.H.M.S.P | Evaluation & Validation, Stakeholder Engagement & Translation | `notebooks/04_stakeholder_validation.ipynb` |

---

## 3. Data Strategy & Provenance

* **Primary Analytical Dataset (Disclosed Proxy)**: [Kaggle E-Commerce Dataset (2023)](https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023) (~2M transaction records). Used as an analytical proxy to learn high-volume association rules and reorder dynamics because the real business has no digitized transaction database.
* **Secondary Reference Dataset**: Sri Lankan Supermarket POS Dataset (~551,639 records across 5 physical branches, 2024–2026) to sanity-check local FMCG buying behaviors (tea, dairy, biscuits, beverages).
* **Stakeholder Evidence**: A signed consultation & consent record and findings review document are maintained in [`docs/`](docs/).

---

## 4. Repository Structure

```text
Supermarket/
│
├── 2026-AI-15_Initial_Report.pdf        # Initial track submission proposal
├── PROJECT_PLAN.md                     # Comprehensive technical plan & RACI matrix
├── README.md                           # Main repository documentation (this file)
├── RULES.md                            # Development, Git workflow, and code hygiene rules
├── requirements.txt                    # Project Python dependencies
├── launch_jupyter.bat                  # 1-click Windows launcher for Chrome
├── launch_jupyter.ps1                  # PowerShell launcher for Chrome
├── .gitignore                          # Git ignore configuration
│
├── docs/                               # Governance & stakeholder evidence
│   ├── README.md
│   ├── 2026-AI-15_Stakeholder_Consent_Letter.docx
│   └── 2026-AI-15_Stakeholder_Data_Addendum.docx
│
├── data/                               # Dataset storage (kept out of git tracking)
│   ├── README.md
│   ├── raw/                            # Immutable raw Kaggle & POS datasets
│   ├── processed/                      # Cleaned parquet tables & basket matrices
│   └── samples/                        # Mini samples for rapid testing (< 5MB)
│
├── notebooks/                          # Sequential experiment and research notebooks
│   ├── README.md                       # Notebook execution roadmap & contracts
│   ├── 01_eda_preprocessing.ipynb      # EDA, cleaning, missing values, sampling
│   ├── 02_market_basket_analysis.ipynb # FP-Growth, Apriori, ECLAT rule mining
│   ├── 03_reorder_prediction.ipynb     # Reorder feature engineering & classification
│   └── 04_stakeholder_validation.ipynb # Local validation & store recommendations
│
├── src/                                # Modular, reusable Python package code
│   ├── README.md
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion, schema validation & downcasting
│   ├── basket_transformer.py           # Pivoting lines to transaction baskets
│   ├── association_engine.py           # Rule mining wrappers & metric filters
│   ├── reorder_classifier.py           # Supervised classifiers & evaluation pipelines
│   └── utils.py                        # Plotting styles, path helpers & figure exports
│
├── models/                             # Saved/serialized model binaries (.joblib)
│   └── README.md
│
└── reports/                            # Generated plots, metrics tables, business summaries
    └── README.md
```

---

## 5. Setup & Installation Guide

Complete this setup once before starting work.

### 5.1 Prerequisites
* **Python 3.10+** (verify with `python --version` on Windows or `python3 --version` on macOS/Linux)
* **Git** installed and configured
* **Google Chrome** (recommended browser for JupyterLab)

---

### 5.2 Step-by-Step Environment Setup

Run these commands from the project root (`e:\Git_Projects\Supermarket`):

#### 1. Create and Activate the Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

> *Your terminal prompt should now show `(.venv)` at the beginning of the line.*

#### 2. Upgrade pip and Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Register the Custom Jupyter Kernel
Registering the virtual environment as a named Jupyter kernel prevents `ModuleNotFoundError` when switching between notebooks:
```bash
python -m ipykernel install --user --name supermarket-env --display-name "Python (Supermarket ML)"
```

**Verify the setup:**
```bash
python -c "import pandas, sklearn, mlxtend; print('All core libraries imported successfully!')"
jupyter kernelspec list
```
*(The output should include `supermarket-env`).*

---

## 6. How to Run Notebooks

### Method A: Open Directly in Google Chrome (Recommended)

#### Option 1: 1-Click Launcher (Fastest)
* In PowerShell terminal:
  ```powershell
  .\launch_jupyter.ps1
  ```
* Or in CMD / File Explorer: Double-click **`launch_jupyter.bat`**.

#### Option 2: Using the Command Line
With your `(.venv)` activated in the terminal:
```powershell
# Launch directly targeting Google Chrome:
jupyter lab --browser="C:/Program Files/Google/Chrome/Application/chrome.exe %s"
```
*(If Chrome is already your default browser, simply typing `jupyter lab` will open it in Chrome automatically).*

---

### Method B: Run Inside the IDE (VS Code / Antigravity Native Editor)

You can run and debug notebooks directly within this IDE without an external browser:

1. In the file explorer, navigate to `notebooks/` and click any notebook (e.g. `01_eda_preprocessing.ipynb`).
2. Look at the top-right corner of the notebook editor:
   * Click **Select Kernel** $\rightarrow$ **Python Environments...**
   * Choose **`.venv (Python 3.14)`** or **`Python (Supermarket ML)`**.
3. Press **Shift + Enter** on individual cells or click **Run All** in the notebook toolbar.

---

## 7. Crucial Concept: Why "ModuleNotFoundError" Happens & How to Avoid It

> ### ⚠️ Why "But I ran pip install" does not automatically fix notebooks:
> Activating `.venv` in PowerShell only affects **that terminal session**.
> When JupyterLab or your IDE opens a notebook, it executes code in an independent process called a **Kernel**.
> 
> * If your notebook is set to the default system `Python 3` kernel, it executes on your base system where `mlxtend` or `sklearn` are NOT installed.
> * **The Fix**: Always check the kernel indicator in the top-right corner of JupyterLab / the IDE. It **must** read:
>   **`Python (Supermarket ML)`** or **`.venv`**.

**The Golden Tip**: Always launch Jupyter *with your virtual environment activated*, or use `python -m jupyterlab`:
```powershell
.\.venv\Scripts\Activate.ps1
python -m jupyterlab
```

---

## 8. Troubleshooting Common Issues

| Symptom | Likely Cause | Exact Fix |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'mlxtend'` (or `pandas`, `sklearn`) | Notebook is running on the default system Python kernel instead of `.venv`. | In JupyterLab top-right: click **Kernel ▸ Change Kernel… ▸ Python (Supermarket ML)**. In IDE: click **Select Kernel ▸ Python Environments ▸ .venv**. |
| `FileNotFoundError: [Errno 2] No such file or directory: '../data/raw/...'` | Working directory mismatch (notebook executed from project root vs. `notebooks/` folder). | Use dynamic path resolution from `src/data_loader.py` or `Path(__file__).resolve()` instead of hardcoded strings. |
| `Execution of cell hangs / MemoryError` | Full 2M transaction table loaded into RAM all at once without type optimization. | Use `create_stratified_sample(df, fraction=0.1)` or downcast numeric types to `category` and `int32` as shown in `src/data_loader.py`. |
| `Scripts cannot be loaded because running scripts is disabled` | Windows PowerShell execution policy restriction. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell window, then rerun `.\.venv\Scripts\Activate.ps1`. |
| Port 8888 is already in use | An older Jupyter server is still running in the background. | Run `jupyter lab --port 8889` or kill the zombie process with `Stop-Process -Name "python" -Force`. |
| "Agentic AI / Python 3" not in kernel list | Kernel was not registered after venv creation. | With `.venv` active, run: `python -m ipykernel install --user --name supermarket-env --display-name "Python (Supermarket ML)"`. |

---

## 9. Development Rules & Git Workflow

All contributors must adhere to the standards outlined in [RULES.md](RULES.md):
* **Branching**: Never push directly to `main`. Create branches like `feature/<name>-<topic>`.
* **Commits**: Follow Conventional Commits format (`feat:`, `fix:`, `docs:`, `chore:`).
* **Data Safety**: Never commit files in `data/raw/` or `data/processed/` (> 25MB). Use `.gitignore`.
* **Reproducibility**: Always enforce `random_state=42` across splits, models, and sampling.

---

## 10. Key References & Documentation
* [PROJECT_PLAN.md](PROJECT_PLAN.md): Detailed analytical roadmap, RACI matrix, and milestone timeline.
* [RULES.md](RULES.md): Git hygiene, commit standards, and error prevention rules.
* [notebooks/README.md](notebooks/README.md): Detailed notebook inputs, outputs, and execution contracts.
* [docs/README.md](docs/README.md): Stakeholder consultation evidence and governance addenda.
