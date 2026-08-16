# Version 1: Deep Learning Baseline

Baseline weekly sales model for 45 stores and 99 departments. Focus: data integrity, multi-input LSTM architecture, and a WMAE benchmark.

## Goals

- Handle missing promotional markdowns without treating missing as zero
- Use entity embeddings for `Store` and `Dept`
- Align training with Weighted Mean Absolute Error (WMAE)
- Identify what limited the LSTM before building V2

## What this version includes

- Binary masking for missing markdown fields
- Multi-input TensorFlow model with entity embeddings
- WMAE evaluation
- Finding: temporal sparsity around holidays was the main LSTM bottleneck

## Stack

Python, pandas, TensorFlow/Keras, NumPy, scikit-learn

## Next

See [Version_2](../Version_2/) for seasonal wave features and the stacked ensemble.
