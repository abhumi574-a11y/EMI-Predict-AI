"""
EMI Prediction Platform — Home
================================
Financial risk assessment platform: EMI eligibility (classification) and
maximum safe EMI amount (regression), built on ML models with MLflow-tracked
experiments, deployable on Streamlit Cloud.
"""

import json

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EMI Prediction Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 EMI Prediction Platform")
st.caption("Financial risk assessment powered by machine learning")

st.markdown(
    """
Welcome! This platform helps financial institutions, FinTech companies, and loan
officers make **data-driven lending decisions** in seconds.

It solves two problems at once:

| Problem | Type | Output |
|---|---|---|
| **EMI Eligibility** | Classification | `Eligible` / `High_Risk` / `Not_Eligible` |
| **Maximum Safe EMI** | Regression | Maximum monthly EMI the applicant can afford (₹) |
"""
)

col1, col2, col3, col4 = st.columns(4)

try:
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    with col1:
        st.metric("Best Classifier", meta["best_classifier"])
        st.metric("Accuracy", f"{meta['classification_results'][meta['best_classifier']]['accuracy']:.1%}")
    with col2:
        st.metric("Best Regressor", meta["best_regressor"])
        st.metric("RMSE", f"₹{meta['regression_results'][meta['best_regressor']]['rmse']:.0f}")
    with col3:
        st.metric("Models trained on", f"{meta['n_rows_trained_on']:,} records")
        st.metric("Input features", "22 raw → 27 engineered")
    with col4:
        st.metric("Classification models", len(meta["classification_results"]))
        st.metric("Regression models", len(meta["regression_results"]))
except FileNotFoundError:
    st.warning(
        "No trained models found yet. Run `python src/train_models.py` first, "
        "or open the **Model Performance** page for setup instructions."
    )

st.divider()

st.subheader("📖 How to use this app")
st.markdown(
    """
Use the sidebar to navigate:

- **🔮 Predict** — enter an applicant's financial profile and get an instant
  eligibility decision + maximum safe EMI recommendation.
- **📊 Data Explorer** — explore the dataset: distributions, correlations, and
  business insights across the 5 EMI scenarios.
- **📈 Model Performance** — compare all trained classification & regression
  models (accuracy, F1, ROC-AUC, RMSE, R²) and see the MLflow tracking setup.
- **🗄️ Data Management** — CRUD operations on applicant records (add, view,
  edit, delete, export).
"""
)

st.divider()
st.subheader("🏗️ Architecture")
st.markdown(
    """
```
Dataset (400K records, 22 features)
        ↓
Data Quality Assessment & Preprocessing
        ↓
Feature Engineering & Exploratory Analysis
        ↓
ML Model Training (≥3 classifiers, ≥3 regressors) & MLflow Tracking
        ↓
Model Evaluation & Best-Model Selection
        ↓
Streamlit Multi-Page Application  ←—— you are here
        ↓
Streamlit Cloud Deployment
```
"""
)

st.info(
    "💡 **Tip:** This demo ships with a synthetic dataset that mirrors the real "
    "schema (5 EMI scenarios, 22 features). Swap in your real data via "
    "`src/data_generator.py` → `data/emi_dataset.csv` and re-run "
    "`src/train_models.py` to retrain on live data.",
    icon="💡",
)
