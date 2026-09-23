# Supermarket Retail Analytics & Predictive Merchandising

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use%20Only-lightgrey.svg)](#)
[![Track: Industry Explorer](https://img.shields.io/badge/Track-Industry%20Explorer-brightgreen.svg)](#)

**Institution**: Sri Lanka Institute of Information Technology (SLIIT)  
**Module**: IT3091 – Machine Learning (Year 3 Semester 1, 2026)  
**Group ID**: `2026-AI-15`  
**Evaluation Track**: **Industry Explorer Track** (External data sources & direct stakeholder consultation)

---

## 1. Project Overview

This project addresses merchandising and inventory replenishment optimization for a real-world, independently operated supermarket. Small retail businesses commonly rely on intuition rather than empirical purchase data, resulting in sub-optimal shelf placement, missed bundling opportunities, and stockouts of high-velocity items.

By leveraging **Unsupervised Association Rule Mining** and **Supervised Reorder Classification**, the project delivers actionable business rules for store layouts, product bundles, and restocking priorities.

### Key Objectives
1. **Product Association & Cross-Selling (Primary Lens)**: Discover frequent item co-occurrences in transactions using **FP-Growth**, benchmarked against **Apriori** and **ECLAT**.
2. **Product-Level Reorder Dynamics (Secondary Lens)**: Model reorder and replenishment frequency using **Logistic Regression**, benchmarked against **Decision Tree** and **Random Forest**.
3. **Stakeholder Grounding & Validation**: Validate findings against direct stakeholder feedback and a secondary Sri Lankan supermarket POS dataset (~551k line items across 5 physical branches).

---

## 2. Team Members & Implementation Roles

| Student ID | Student Name | Core Implementation Responsibility |
| :--- | :--- | :--- |
| **IT24103684** | Pehesara W.A.C | Data Collection, Ingestion, Preprocessing & EDA |
| **IT24103866** | Obesekara S.O.K.D | Product Association Mining (FP-Growth, Apriori, ECLAT) |
| **IT24103792** | Peiris W.S.V | Reorder Dynamics Feature Engineering & Supervised Modeling |
| **IT24104192** | Weerasooriya W.H.M.S.P | Evaluation & Validation, Stakeholder Engagement & Translation |

---

## 3. Data Strategy & Provenance

* **Primary Analytical Dataset (Disclosed Proxy)**: [Kaggle E-Commerce Dataset (2023)](https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023) (~2M transaction records). Used as an analytical proxy because the stakeholder's small physical shop does not maintain a digitized POS database.
* **Secondary Reference Dataset**: Sri Lankan Supermarket POS Dataset (~551,639 records, 2024–2026) to test local retail validity (staple FMCG: tea, dairy, biscuits).
* **Stakeholder Evidence**: Signed stakeholder consultation and consent letter documented in `docs/`.

---

## 4. Repository Structure

```text
Supermarket/
│
├── 2026-AI-15_Initial_Report.pdf        # Formal submission proposal
├── PROJECT_PLAN.md                     # Detailed technical project plan & specification
├── RULES.md                            # Development, Git workflow, and code hygiene rules
├── README.md                           # Main repository documentation (this file)
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git ignore configuration
│
├── docs/                               # Consultation records, consent forms, data addenda
│   ├── README.md
│   ├── 2026-AI-15_Stakeholder_Consent_Letter.docx
│   └── 2026-AI-15_Stakeholder_Data_Addendum.docx
│
├── data/                               # Dataset storage (kept out of git tracking)
│   ├── README.md
│   ├── raw/                            # Immutable raw data files
│   ├── processed/                      # Cleaned parquet/csv files & basket matrices
│   └── samples/                        # Mini samples for testing
│
├── notebooks/                          # Sequential experiment and research notebooks
│   ├── README.md
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_market_basket_analysis.ipynb
│   ├── 03_reorder_prediction.ipynb
│   └── 04_stakeholder_validation.ipynb
│
├── src/                                # Modular, reusable Python package code
│   ├── README.md
│   ├── __init__.py
│   ├── data_loader.py
│   ├── basket_transformer.py
│   ├── association_engine.py
│   ├── reorder_classifier.py
│   └── utils.py
│
├── models/                             # Saved/serialized model binaries (.pkl, .joblib)
│   └── README.md
│
└── reports/                            # Generated plots, metrics tables, business summaries
    └── README.md
```

---

## 5. Quick Start & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd Supermarket
   ```

2. **Create and activate a virtual environment**:
   * Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Register kernel for Jupyter**:
   ```bash
   python -m ipykernel install --user --name supermarket-env --display-name "Python (Supermarket)"
   ```

---

## 6. Contribution & Team Guidelines

Please review [RULES.md](RULES.md) before making commits or pushing code. Key rules include:
* **Branching Model**: Never push directly to `main` without testing. Use branch format `feature/<member>-<task>`.
* **Commit Conventions**: Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
* **Data Hygiene**: Never commit large raw data or sensitive files. Check `.gitignore`.
* **Reproducibility**: Always specify random states (`random_state=42`) and write modular code in `src/`.

---

## 7. Key Documentation Links
* [PROJECT_PLAN.md](PROJECT_PLAN.md): Full technical specification, RACI matrix, and timeline.
* [RULES.md](RULES.md): Development hygiene, Git rules, and error prevention guidelines.
* [docs/README.md](docs/README.md): Stakeholder consultation and governance records.
