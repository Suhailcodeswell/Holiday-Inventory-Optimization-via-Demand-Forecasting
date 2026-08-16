# Walmart Demand Forecasting

Weekly sales forecasting for 45 Walmart stores and 99 departments. The work moved from a deep learning baseline (V1) to a stacked ensemble (V2) built for a short, two-year history with strong holiday seasonality.

## Results

| Model | Version | WMAE (lower is better) |
| --- | --- | ---: |
| Baseline LSTM | V1 | 5506.86 |
| Optimized LSTM | V2 | 2819.43 |
| Stacked ensemble (70% XGBoost / 30% LSTM) | V2 | 2131.56 |

V2 cut LSTM error by about 48% versus the V1 baseline.

## Approach

- Entity embeddings for store and department cardinality
- 21-day seasonal influence waves instead of binary holiday flags
- Log-space target scaling for LSTM training
- Stacked XGBoost + LSTM ensemble with SHAP for feature importance

## Repository

| Folder | Contents |
| --- | --- |
| [Version_1](./Version_1/) | Baseline LSTM, data handling, WMAE setup |
| [Version_2](./Version_2/) | Seasonal waves, stacking, final models |
| [Data](./Data/) | Training and test CSVs |
| [Academic_Documents](./Academic_Documents/) | Course planning materials |

## Stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, XGBoost, SHAP, Matplotlib, Seaborn

## How to run

1. Place `train.csv`, `test.csv`, `stores.csv`, and `features.csv` in `Data/`.
2. Follow [Version_1/README.md](./Version_1/README.md) for the baseline.
3. Follow [Version_2/README.md](./Version_2/README.md) for the final ensemble.

Update any notebook paths that point to `/content/` if you are running locally.

## Course

MGTA 611, Business Application of Artificial Intelligence, Wilfrid Laurier University
