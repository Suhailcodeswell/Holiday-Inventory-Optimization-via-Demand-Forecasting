# Project Walmart: Holiday Sales Forecasting (Version 2)

### *Subtitle: Seasonal waves, log targets & stacked models*

## Project overview

Version 2 builds on Version 1 with a focus on **faculty feedback**: holiday effects are **sparse in calendar time** (only a few Thanksgiving/Christmas windows per store in the sample), so the model needs **richer holiday structure** and **stable targets**, not only more years of raw data.

V2 introduces **holiday influence curves** (multi-week ramps), **macro interactions**, **log-transformed sales** for neural training, an **XGBoost + LSTM ensemble**, and a **final XGBoost + SHAP** pass on a **fixed 13-feature** numeric set.

## Version 2 objectives

- **Mitigate temporal sparsity of holidays:** Replace reliance on a single binary `IsHoliday` flag with **continuous “influence” features** before each major event.
- **Stabilize learning:** Train on **`log1p(Weekly_Sales)`** and evaluate in **dollars** via `expm1` for WMAE.
- **Capture economics × holidays:** Add interaction features (e.g. Christmas influence × unemployment / CPI).
- **Combine strengths:** **Stack** XGBoost (tabular) and LSTM (nonlinear) predictions; retrain XGBoost with **explicit Super Bowl** influence in the feature set and run **SHAP** for the final report.

---

## Technical implementation

### 1. Holiday influence waves (21-day taper)

For **Super Bowl**, **Thanksgiving**, and **Christmas**, V2 computes:

- **`Days_to_{holiday}`** — days until the next listed occurrence in the training years.
- **`{holiday}_Influence`** — a value in \([0,1]\) that **ramps up** as the event approaches (commented in code as a **21-day** window). This gives the model **many more weeks** with nonzero “holiday pressure” than a single labeled holiday row—directly addressing **“not enough holiday observations”** in a short panel.

### 2. Preprocessing (aligned with V1)

- **Binary masking** for `MarkDown1`–`MarkDown5` + fill with `0`.
- **Returns:** negative `Weekly_Sales` clipped to `0`.
- **`Sales_Log = log1p(Weekly_Sales)`** for modeling.
- **`CPI` / `Unemployment`** forward-filled; **`Type`** label-encoded; **`IsHoliday`** integer.

### 3. Feature engineering beyond V1

- **`Lag_52`:** `groupby(['Store','Dept'])['Weekly_Sales'].shift(52)` — same week last year, per series.
- **`Holiday_Unemployment_Impact`** = `Christmas_Influence × Unemployment`
- **`Holiday_CPI_Impact`** = `Christmas_Influence × CPI` (in the data pipeline; primary LSTM block uses a curated 12-feature set—see notebook).
- **`Fuel_Price_Trend`:** within-store first difference of `Fuel_Price`.

### 4. Models

- **LSTM branch:** Store/Dept **embeddings** + LSTM on continuous tensor `(batch, 1, n_features)`; **64** LSTM units; **MAE on log sales**; **sample weights** (5× on holiday weeks) for alignment with WMAE in dollar space at evaluation time.
- **XGBoost:** Trained on **log** targets with same weights; predictions converted with **`expm1`**.
- **Ensemble:** Blend **70% XGBoost + 30% LSTM** in dollar space (weights can be tuned).
- **Final polish block:** Retrain XGBoost on **13 numeric columns** including **`SuperBowl_Influence`**, force **`float`** dtypes for SHAP, regenerate **ensemble** and **SHAP summary** for documentation.

### 5. Evaluation

Same **WMAE** definition as V1: weights \(5\) on `IsHoliday` weeks, \(1\) otherwise, computed on **dollar** sales after reversing the log transform for model outputs.

---

## Files in this folder

| File | Description |
|------|-------------|
| `Demand_Forecasting_V2.ipynb` | Primary V2 notebook (Colab export). |
| `demand_forecasting_v2.py` | Same logic as a runnable Python script. |
| `README.md` | This documentation. |

**Note:** The exported script uses Colab paths (`/content/train.csv`). For local runs, change to `Data/train.csv` (and same for `stores.csv`, `features.csv`) or run from a folder where those files exist.

---

## Relation to Version 1

| Aspect | V1 | V2 |
|--------|----|-----|
| Holiday signal | Binary + days-to | **Influence curves** + interactions |
| Target | Levels | **Log1p** for NN/XGB training |
| Models | XGB + LSTM + tune + SHAP | XGB + LSTM + **ensemble** + **final XGB + SHAP (13 features)** |
| Test CSV | Loaded in V1 pipeline | Not required in V2 script (focus on validation design) |

Use **`../Version_1/`** for the full baseline narrative; use **this folder** for the **refined strategy** and final-course storyline.
