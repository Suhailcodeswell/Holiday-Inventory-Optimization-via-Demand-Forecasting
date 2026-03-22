# 🛒 Walmart Retail Demand Forecasting: An Iterative AI Approach

### *Solving the $1.7T Inventory Distortion Problem with Hybrid Deep Learning*

## 🌟 Project Overview

This repository contains a two-phase development cycle for predicting weekly sales across 45 Walmart stores and 99 departments. The project moves from an initial **Deep Learning Baseline (V1)** to a sophisticated **Stacked Ensemble System (V2)** designed to overcome the "Small Data" challenges of the 2-year Walmart dataset.

### **The Core Challenge**

Predicting retail demand is notoriously difficult due to **high cardinality** (thousands of store-dept combinations) and **extreme seasonality** (holidays like Black Friday and Christmas). Traditional models often fail to capture the "ramp-up" period before these events when data is limited to only two years.

---

## 📂 Repository Structure

### **📁 [Version_1: The Baseline Exploration](./Version_1/)**

*Focus: Architecture and Data Integrity.*

* Implemented **Binary Masking** for missing Markdown data.
* Developed a **Multi-Input Functional API** (TensorFlow) using **Entity Embeddings** for Stores/Depts.
* Established the **Weighted Mean Absolute Error (WMAE)** benchmark.
* **Key Result:** Identified "Temporal Sparsity" as the primary bottleneck for LSTM performance.

### **📁 [Version_2: The Production Solution](./Version_2/)**

*Focus: Feature Engineering and Model Stacking.*

* **Innovation:** Engineered **Seasonal Influence Waves** (21-day ramp-ups) to solve data scarcity.
* **Innovation:** Implemented **Log-Space Target Scaling** for smoother convergence.
* **Architecture:** A **Stacked Ensemble** (70% XGBoost / 30% LSTM) that merges tree-based logic with temporal neurons.
* **Key Result:** Reduced LSTM error by **48%** and achieved a final **WMAE of 2131.56**.

---

## 🛠 Technical Stack

* **Languages:** Python (Pandas, NumPy, Scikit-Learn)
* **Deep Learning:** TensorFlow / Keras (LSTM, Embedding Layers)
* **Machine Learning:** XGBoost (Gradient Boosting)
* **Interpretability:** SHAP (Shapley Additive Explanations)
* **Visualization:** Matplotlib, Seaborn

---

## 📈 Methodology Highlights

### **1. Addressing the "Holiday Effect"**

Instead of using simple binary flags, we treated holidays as **Seasonal Waves**. This allowed our models to learn the shopping momentum leading up to major events, effectively tripling our high-importance training signals.

### **2. Why an Ensemble?**

We discovered that XGBoost was excellent at handling historical averages, while the LSTM was better at finding non-linear patterns in economic data (CPI, Unemployment). By **Stacking** them, we canceled out individual model biases.

---

## 🏆 Final Performance Summary

| Model | Version | WMAE (Lower is Better) |
| :--- | :--- | :--- |
| **Baseline LSTM** | V1 | 5506.86 |
| **Optimized LSTM** | V2 | 2819.43 |
| **Final Stacked Ensemble** | **V2** | **2131.56** |

---

## 🚀 How to Use

1. **Version 1:** Explore the foundational notebooks to see the initial architecture and data cleaning — see [`Version_1/README.md`](./Version_1/README.md).
2. **Version 2:** Review the final production-ready scripts and the SHAP analysis for feature importance — see [`Version_2/README.md`](./Version_2/README.md).
3. **Data:** Place `train.csv`, `test.csv`, `stores.csv`, and `features.csv` in the **`Data/`** folder (as in this repo), or update `read_csv` paths in each script/notebook to match your layout. Version 1 scripts often expect CSVs in the working directory unless paths are adjusted; Version 2 Colab exports may use `/content/` — change to `Data/...` for local runs.

---

**Course:** MGTA 611 — Business Application of Artificial Intelligence · Wilfrid Laurier University
