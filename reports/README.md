# Analytical Reports & Generated Artifacts

This directory stores output figures, evaluation plots, rulebook exports, and benchmark tables produced by the analysis.

---

## Directory Organization

```text
reports/
├── figures/              # Exported publication-ready charts (PNG / PDF at 300 DPI)
│   ├── basket_size_distribution.png
│   ├── top_departments_frequency.png
│   ├── roc_curve_comparison.png
│   └── feature_importance_reorder.png
├── tables/               # Exported CSVs of rules, metrics, and benchmarks
│   ├── top_association_rules.csv
│   └── model_performance_comparison.csv
└── business_rules/       # Curated guides tailored for stakeholder review
    ├── store_layout_recommendations.md
    └── restocking_priority_schedule.md
```

---

## Guidelines
* Always generate figures using `src.utils.save_figure()` to enforce consistent styling and resolution.
* Ensure all tables exported here use clear column naming without internal abbreviations.
