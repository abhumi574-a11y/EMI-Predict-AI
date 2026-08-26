"""Shared helpers for loading models/data, cached across Streamlit pages."""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))


@st.cache_resource
def load_classifier():
    return joblib.load(os.path.join(MODELS_DIR, "best_classifier.joblib"))


@st.cache_resource
def load_regressor():
    return joblib.load(os.path.join(MODELS_DIR, "best_regressor.joblib"))


@st.cache_data
def load_metadata():
    with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_sample_data(n=5000):
    path = os.path.join(DATA_DIR, "emi_dataset.csv")
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return df


def models_available() -> bool:
    required = ["preprocessor.joblib", "best_classifier.joblib", "best_regressor.joblib", "model_metadata.json"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def data_available() -> bool:
    return os.path.exists(os.path.join(DATA_DIR, "emi_dataset.csv"))


def predict_applicant(applicant: dict):
    """applicant: dict of the 22 raw input fields -> returns (eligibility, probs, max_emi)."""
    pre = load_preprocessor()
    clf = load_classifier()
    reg = load_regressor()

    df = pd.DataFrame([applicant])
    X = pre.transform(df)

    eligibility = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    proba_dict = dict(zip(clf.classes_, proba))
    max_emi = reg.predict(X)[0]

    return eligibility, proba_dict, max_emi
