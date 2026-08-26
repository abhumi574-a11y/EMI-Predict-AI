import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model_utils import data_available, load_sample_data  # noqa: E402

st.set_page_config(page_title="Data Explorer | EMI Platform", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.caption("Exploratory analysis of the EMI dataset across scenarios, demographics, and risk factors.")

if not data_available():
    st.error(
        "No dataset found at `data/emi_dataset.csv`. Run "
        "`python src/data_generator.py --rows 400000 --out data/emi_dataset.csv` first."
    )
    st.stop()

df = load_sample_data(n=8000)
st.caption(f"Showing analysis on a random sample of {len(df):,} records (full dataset may be larger).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records analyzed", f"{len(df):,}")
k2.metric("Avg. monthly salary", f"₹{df['monthly_salary'].mean():,.0f}")
k3.metric("Avg. credit score", f"{df['credit_score'].mean():.0f}")
k4.metric("Eligible rate", f"{(df['emi_eligibility'] == 'Eligible').mean():.1%}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Eligibility & Scenarios", "📈 Correlations", "👥 Demographics", "🔎 Raw Data"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("EMI Eligibility Distribution")
        st.bar_chart(df["emi_eligibility"].value_counts())
    with c2:
        st.subheader("Records by EMI Scenario")
        st.bar_chart(df["emi_scenario"].value_counts())

    st.subheader("Eligibility Rate by EMI Scenario")
    pivot = pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index") * 100
    st.dataframe(pivot.round(1).astype(str) + "%", use_container_width=True)

    st.subheader("Max Monthly EMI by Scenario")
    st.bar_chart(df.groupby("emi_scenario")["max_monthly_emi"].mean())

with tab2:
    st.subheader("Correlation with Max Monthly EMI")
    numeric_cols = [
        "age", "monthly_salary", "years_of_employment", "credit_score",
        "bank_balance", "emergency_fund", "current_emi_amount",
        "requested_amount", "requested_tenure", "max_monthly_emi",
    ]
    corr = df[numeric_cols].corr()["max_monthly_emi"].drop("max_monthly_emi").sort_values()
    st.bar_chart(corr)

    st.subheader("Full Correlation Matrix")
    st.dataframe(df[numeric_cols].corr().round(2), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Credit Score vs. Eligibility")
        st.bar_chart(df.groupby("emi_eligibility")["credit_score"].mean())
    with c2:
        st.subheader("Monthly Salary vs. Eligibility")
        st.bar_chart(df.groupby("emi_eligibility")["monthly_salary"].mean())

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Age Distribution")
        age_bins = pd.cut(df["age"], bins=[24, 30, 40, 50, 61]).astype(str)
        st.bar_chart(age_bins.value_counts().sort_index())
    with c2:
        st.subheader("Education Level Distribution")
        st.bar_chart(df["education"].value_counts())

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Employment Type Distribution")
        st.bar_chart(df["employment_type"].value_counts())
    with c4:
        st.subheader("House Type Distribution")
        st.bar_chart(df["house_type"].value_counts())

    st.subheader("Average Salary by Education Level")
    st.bar_chart(df.groupby("education")["monthly_salary"].mean())

with tab4:
    st.subheader("Sample Records")
    st.dataframe(df.head(200), use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download this sample as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="emi_sample.csv",
        mime="text/csv",
    )
