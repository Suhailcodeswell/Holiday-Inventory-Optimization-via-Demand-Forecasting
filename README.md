# Holiday Inventory Optimization via Demand Forecasting

**Course:** MGTA 611 — Business Application of Artificial Intelligence  
**Theme:** Use **machine learning** and **deep learning** to forecast **weekly department-level sales** at Walmart-scale retail, with emphasis on **holiday weeks** where errors are most costly.

---

## What this project is

Retail demand is volatile around **Thanksgiving, Christmas, Super Bowl**, and other labeled holiday periods. Wrong inventory leads to **stockouts** (lost sales, poor customer experience) or **overstock** (markdowns, margin loss). This repository implements a **data-driven forecasting pipeline** on the **Walmart recruiting store-sales** style dataset: weekly sales for many **store × department** time series, plus **economic and promotional features**.

We treat the problem as **supervised regression**: predict `Weekly_Sales` from past sales, calendar features, store metadata, fuel/CPI/unemployment, and markdown/promotion fields—then evaluate with **Weighted Mean Absolute Error (WMAE)** that **up-weights holiday weeks**.

---

## What we did (high level)

1. **Data integration** — Merge sales (`train.csv` / `test.csv`) with `features.csv` (macro, weather, markdowns) and `stores.csv` (format type, size).
2. **Missing promotions** — Markdown columns are often missing early in the sample; we use **binary masks** so “missing” is not confused with “no promotion.”
3. **Baselines & challengers** — **XGBoost** with holiday-weighted training; **neural model** with **entity embeddings** for `Store` and `Dept` and an **LSTM** branch on continuous features.
4. **Metric** — **WMAE** with weight **5×** on holiday weeks and **1×** otherwise, matching the business priority to be accurate when it matters most.
5. **Iteration (Version 2)** — Address **sparse holiday signal** with **holiday influence features**, **log-transformed targets**, **economic × holiday interactions**, and a **stacked ensemble** plus **SHAP** on a refined feature set for interpretation.

---

## Repository layout

```
├── README.md                 ← You are here (project map)
├── Data/                     ← CSVs: train, test, features, stores
├── Version_1/               ← Baseline pipeline: notebook + script + README
├── Version_2/               ← Enhanced strategy: notebook + script + README
├── MGTA 611 Term Project Proposal.pdf
├── Project Planning .pdf
└── … (other course documents)
```

| Path | Purpose |
|------|---------|
| **`Version_1/`** | **Proof-of-concept:** binary markdown masks, holiday proximity, lag-52, XGBoost + embedding LSTM, WMAE, SHAP on XGBoost. |
| **`Version_2/`** | **Production strategy:** holiday “influence” waves, log targets, interaction features, ensemble, final XGBoost + SHAP on 13 numeric features. |

---

## Data (short reference)

| File | Role |
|------|------|
| `Data/train.csv` | Store, Dept, Date, **Weekly_Sales**, IsHoliday |
| `Data/test.csv` | Same keys without `Weekly_Sales` (scoring / deployment) |
| `Data/features.csv` | Store, Date: temperature, fuel, markdowns, CPI, unemployment, IsHoliday |
| `Data/stores.csv` | Store: Type, Size |

---

## How to run (quick)

- **Python 3.10+** recommended. Install **pandas**, **numpy**, **scikit-learn**, **xgboost**, **tensorflow**, **matplotlib**, **seaborn**, **shap** (for interpretability cells).
- Place CSVs under `Data/` and either:
  - Run notebooks from the repo root and adjust `read_csv` paths to `Data/train.csv`, etc., or  
  - Copy/link CSVs next to the script if the script expects `train.csv` in the current directory (see each version’s README).

---

## What we achieved

- A **reproducible** two-stage story: **V1** establishes **embeddings + WMAE + interpretable tree baseline**; **V2** responds to **faculty feedback** on **holiday data sparsity** with **richer holiday features** and a **combined modeling strategy**.
- **Business-aligned evaluation** (holiday-weighted error), not only raw MAE.
- **Interpretability** via **SHAP** on gradient-boosted models to support **inventory and markdown** discussion in the report and presentation.

---

## Team & acknowledgments

Project work by the MGTA 611 group (see proposal PDF for names). Course materials reference **Michael Blair**, Wilfrid Laurier University.

**License / use:** Academic course project; dataset subject to original **Walmart / Kaggle** competition terms if used outside class.

---

## Where to read next

- **Version 1 details:** [`Version_1/README.md`](Version_1/README.md)  
- **Version 2 details:** [`Version_2/README.md`](Version_2/README.md)
