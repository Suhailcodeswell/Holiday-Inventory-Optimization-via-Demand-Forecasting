# Project Walmart: Holiday Sales Forecasting (Version 1)

### *Subtitle: Deep learning baseline & high-cardinality exploration*

## Project overview

Version 1 is the foundational research phase for predicting **weekly sales** across **45 Walmart stores** and **99 departments**. The main challenges are **high-cardinality** store–department combinations and **sparse promotional markdowns** that interact with **holiday weeks**, where forecast error is especially costly.

## Version 1 objectives

- **Data integrity:** Handle **missing-not-at-random (MNAR)** patterns in promotional markdowns without treating “missing” the same as “zero promotion.”
- **Architecture:** Build a **multi-input** neural network using **entity embeddings** for categorical variables (`Store`, `Dept`).
- **Metric alignment:** Implement **Weighted Mean Absolute Error (WMAE)** so errors in **holiday weeks** count more than in normal weeks.

---

## Technical implementation

### 1. Sparse markdowns (binary masking)

A large share of `MarkDown1`–`MarkDown5` values are missing (especially in early periods). In V1 we use **binary masking**:

- For each markdown column, add a flag such as `MarkDown1_missing` (1 if missing, 0 otherwise).
- Fill missing markdown values with `0` for the raw feature so the model can still use numeric inputs.
- The mask lets the model learn that **“no data recorded”** is different from **“promotion was zero.”**

### 2. Hybrid multi-input architecture (TensorFlow / Keras)

V1 uses a **functional API** model with three inputs:

- **Categorical branch:** **Embedding** layers map `Store` and `Dept` to **10-dimensional** vectors so similar entities can sit closer in learned space than one-hot encoding would allow.
- **Continuous branch:** All other numeric features (economic indicators, lags, holiday proximity, etc.) are shaped as `(batch, 1, n_features)` and passed through an **LSTM** layer (sequence length 1 in this implementation—see notebook for details).
- **Fusion:** Embeddings are flattened, concatenated with the LSTM output, then passed through **Dense** layers with **Dropout (0.2)** for regularization.

A separate **XGBoost** regressor (MAE-oriented objective with **sample weights**) provides a strong **tabular baseline** before the LSTM “challenger.”

### 3. Evaluation: weighted MAE (WMAE)

Primary metric:

$$
\mathrm{WMAE} = \frac{\sum_{i=1}^{n} w_i \, |y_i - \hat{y}_i|}{\sum_{i=1}^{n} w_i}
$$

- **Non-holiday weeks:** \(w = 1\)
- **Holiday weeks** (per dataset `IsHoliday`): \(w = 5\)

Both **XGBoost** (`sample_weight`) and **LSTM** (`sample_weight` on MAE loss) use these weights so training aligns with business priorities.

### 4. Other preprocessing (summary)

- Negative `Weekly_Sales` (returns) clipped to `0` in training.
- `CPI` and `Unemployment` forward-filled where needed.
- `Type` (store format) label-encoded; `IsHoliday` cast to integer.
- **Holiday proximity:** `Days_to_SuperBowl`, `Days_to_Christmas`, `Days_to_Thanksgiving`.
- **Lag feature:** `Lag_52` (same store–department, week offset one year) via merge to historical sales.
- Chronological **80% / 20%** train–validation split after sorting by date.
- **SHAP** (TreeExplainer on XGBoost) for interpretability of drivers of predicted sales.

---

## Results & faculty feedback (reference)

Example validation figures reported from this pipeline (your environment may differ slightly):

| Model              | WMAE (validation) |
|--------------------|-------------------:|
| XGBoost baseline   | ~1737.79          |
| LSTM (tuned)       | ~3237.66          |

**Interpretation:** The tree baseline can outperform the first LSTM configuration when the sequence is effectively one timestep wide and holiday signal is sparse—this motivated **Version 2** (holiday “influence” features, log target, interactions, ensemble, and refined XGBoost + SHAP).

**Faculty note:** With only a few calendar years in the sample, **raw holiday labels** are sparse; deep models need either more **explicit holiday structure** (see V2) or careful validation design. V1 remains valuable as the **end-to-end baseline + interpretability** track.

---

## Files in this folder

| File | Description |
|------|-------------|
| `611 Script.ipynb` | Primary research notebook (Colab export). |
| `611_script.py` | Same logic as a runnable Python script. |
| `README.md` | This documentation. |

**Data:** Training features live in the repo root `Data/` folder (`train.csv`, `test.csv`, `features.csv`, `stores.csv`). When running locally, set the working directory or paths so `train.csv` resolves (or update paths to `Data/train.csv`).

---

## Team & course

**MGTA 611 – Business Application of Artificial Intelligence** — Holiday inventory & demand forecasting (Walmart store sales).  
Version 1 = **proof-of-concept**: embeddings + WMAE + XGBoost/LSTM + SHAP.

For the **production-oriented** iteration (holiday waves, log sales, stacking, final SHAP on 13 features), see **`../Version_2/`**.
