# 🚀 Project Walmart: Holiday Sales Forecasting (Version 2)

### *Sub-title: Overcoming Data Scarcity with Seasonal Waves & Ensemble Stacking*

## 📖 Project Overview

Version 2 (V2) represents the transition from an experimental baseline to a **Production-Ready Forecasting System**. This version was specifically engineered to solve the **"Small Data" problem** inherent in the 2-year Walmart dataset, where traditional LSTMs struggle to identify holiday patterns with only two observed cycles.

## 🛠 Strategic Innovations (The V2 Pivot)

### **1. Seasonal Influence Waves (The Professor's Fix)**

To address faculty feedback regarding limited holiday data, we moved away from binary "0 or 1" holiday flags.

* **The Strategy:** We engineered **21-day "Ramp-up" Windows** for Christmas, Thanksgiving, and the Super Bowl.
* **The Impact:** This tripled the amount of high-importance training data, allowing the model to learn the *momentum* of holiday shopping rather than just a single-day spike.

### **2. Log-Space Target Scaling**

Deep Learning models often struggle with high-variance targets (like sales ranging from $\$0$ to $\$200,000$).

* **The Strategy:** We transformed the target variable using `np.log1p`.
* **The Impact:** This stabilized the LSTM's loss function, focusing the model on **percentage growth** rather than raw dollar amounts, leading to a much smoother convergence.

### **3. Economic Interaction Terms**

We recognized that holidays don't happen in a vacuum.

* **The Strategy:** Created interaction features like `Holiday_Unemployment_Impact`.
* **The Impact:** This allowed the model to understand how the macro-economy (Unemployment/CPI) dampens or amplifies the "Seasonal Wave" effect.

---

## 🧠 Model Architecture: The Stacked Ensemble

V2's final output is not a single model, but a **Hybrid Stacked Ensemble**:

1. **XGBoost (V2):** Captures the rigid, tree-based logic of historical averages.
2. **LSTM (V2):** Captures the non-linear, temporal relationships in the economic and seasonal data.
3. **The Blend:** A **70/30 weighted average** that balances the strengths of both, significantly reducing individual model bias.

---

## 📊 Final Results & Performance

* **Baseline LSTM (V1):** 5506.86 WMAE
* **Optimized LSTM (V2):** 2819.43 WMAE (**+48.8% Improvement**)
* **Final Stacked Ensemble:** **2131.56 WMAE**

### **Model Explainability (SHAP)**

Using SHAP values, we confirmed that while `Lag_52` (Year-over-Year) remains the primary driver, our new `Fuel_Price_Trend` and `Holiday_Influence` features rank in the top 10 most impactful variables.

---

## 📂 Files in this Folder

* `Demand_Forecasting_V2.ipynb`: The high-performance ensemble notebook.
* `Demand_Forecasting_V2.py`: The Python production script.
* `README.md`: This documentation.
