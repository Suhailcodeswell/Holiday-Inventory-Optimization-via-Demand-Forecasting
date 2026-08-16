# Version 2: Stacked Ensemble

Production-oriented forecasting pass for the same Walmart weekly sales task. Built to handle a short two-year history and sparse holiday cycles.

## Changes from V1

1. **Seasonal influence waves:** 21-day ramps into Christmas, Thanksgiving, and Super Bowl instead of binary holiday flags.
2. **Log-space targets:** `np.log1p` on sales for more stable LSTM training.
3. **Economic interactions:** Features that combine holidays with CPI and unemployment.
4. **Stacked ensemble:** 70% XGBoost + 30% LSTM.

## Results

| Model | WMAE |
| --- | ---: |
| Baseline LSTM (V1) | 5506.86 |
| Optimized LSTM (V2) | 2819.43 |
| Final stacked ensemble | 2131.56 |

SHAP analysis shows year-over-year lag features as primary drivers, with the seasonal wave features contributing meaningful holiday signal.

## Stack

Python, pandas, TensorFlow/Keras, XGBoost, SHAP

## How to run

1. Point scripts and notebooks at the CSVs in `../Data/`.
2. Run the V2 forecasting notebook or script in this folder.
3. Review SHAP outputs for feature importance.

See the root [README](../README.md) for the full project summary.
