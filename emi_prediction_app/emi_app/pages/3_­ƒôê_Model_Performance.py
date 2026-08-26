import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model_utils import load_metadata, models_available  # noqa: E402

st.set_page_config(page_title="Model Performance | EMI Platform", page_icon="📈", layout="wide")
st.title("📈 Model Performance & MLflow Tracking")
st.caption("Compare every trained model and see how MLflow experiment tracking is wired in.")

if not models_available():
    st.error(
        "No trained models found. From the project root, run:\n\n"
        "```bash\npython src/data_generator.py --rows 400000\n"
        "python src/train_models.py --sample 100000\n```"
    )
    st.stop()

meta = load_metadata()

st.subheader("🏆 Selected Models for Production")
c1, c2 = st.columns(2)
c1.success(f"**Best Classifier:** {meta['best_classifier']}")
c2.success(f"**Best Regressor:** {meta['best_regressor']}")
st.caption(
    f"Trained on {meta['n_rows_trained_on']:,} records · "
    f"{'XGBoost available' if meta['used_xgboost'] else 'XGBoost not installed — used GradientBoosting as substitute (install xgboost for the full spec)'}"
)

st.divider()

tab1, tab2 = st.tabs(["🎯 Classification Models", "📉 Regression Models"])

with tab1:
    st.subheader("EMI Eligibility — Classification Models")
    clf_df = pd.DataFrame(meta["classification_results"]).T
    clf_df.index.name = "Model"
    st.dataframe(
        clf_df.style.highlight_max(subset=["accuracy", "f1_score", "roc_auc"], color="#c6f6d5")
        .format({c: "{:.4f}" for c in clf_df.columns if c != "train_time_sec"})
        .format({"train_time_sec": "{:.1f}s"}),
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Accuracy by model")
        st.bar_chart(clf_df["accuracy"])
    with c2:
        st.caption("ROC-AUC by model")
        st.bar_chart(clf_df["roc_auc"])
    st.info(
        "**Metric guide:** Accuracy = overall correctness · Precision = of predicted "
        "positives, how many were right · Recall = of actual positives, how many were "
        "found · F1 = harmonic mean of precision/recall · ROC-AUC = ability to rank "
        "classes correctly (closer to 1 is better)."
    )

with tab2:
    st.subheader("Maximum Monthly EMI — Regression Models")
    reg_df = pd.DataFrame(meta["regression_results"]).T
    reg_df.index.name = "Model"
    st.dataframe(
        reg_df.style.highlight_min(subset=["rmse", "mae"], color="#c6f6d5")
        .highlight_max(subset=["r2"], color="#c6f6d5")
        .format({c: "{:.4f}" for c in reg_df.columns if c != "train_time_sec"})
        .format({"train_time_sec": "{:.1f}s"}),
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("RMSE by model (₹, lower is better)")
        st.bar_chart(reg_df["rmse"])
    with c2:
        st.caption("R² by model (higher is better)")
        st.bar_chart(reg_df["r2"])
    st.info(
        "**Metric guide:** RMSE/MAE = average prediction error in ₹ (lower is better) · "
        "R² = share of variance explained (closer to 1 is better) · MAPE = average % error."
    )

st.divider()
st.subheader("🔬 MLflow Experiment Tracking")
st.markdown(
    """
Every model run in `src/train_models.py` is logged to MLflow when it's installed —
parameters, metrics (accuracy/F1/ROC-AUC or RMSE/MAE/R²/MAPE), and the fitted model
artifact, organized under the **`EMI_Prediction_Platform`** experiment.

**To view the MLflow dashboard locally:**
```bash
pip install mlflow
python src/train_models.py --sample 100000   # logs runs to ./mlruns
mlflow ui                                     # open http://localhost:5000
```

From the dashboard you can compare every classification and regression run side by
side, inspect hyperparameters, and promote a model to the **Model Registry** for
versioned production deployment.
"""
)
