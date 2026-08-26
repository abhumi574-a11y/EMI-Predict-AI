"""
Feature Engineering
====================
Turns the raw 22 input columns into a model-ready feature matrix:
- Derived financial ratios (debt-to-income, expense-to-income, affordability)
- Risk scoring features
- Categorical encoding
- Numerical scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

CATEGORICAL_COLS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]

RAW_NUMERIC_COLS = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven ratio & risk features. Returns a new dataframe."""
    d = df.copy()

    total_expenses = (
        d["monthly_rent"]
        + d["school_fees"]
        + d["college_fees"]
        + d["travel_expenses"]
        + d["groceries_utilities"]
        + d["other_monthly_expenses"]
        + d["current_emi_amount"]
    )
    salary_safe = d["monthly_salary"].replace(0, np.nan)

    d["total_monthly_expenses"] = total_expenses
    d["disposable_income"] = d["monthly_salary"] - total_expenses
    d["debt_to_income_ratio"] = (d["current_emi_amount"] / salary_safe).fillna(0)
    d["expense_to_income_ratio"] = (total_expenses / salary_safe).fillna(0)
    d["affordability_ratio"] = (d["disposable_income"] / salary_safe).fillna(0)
    d["savings_ratio"] = (d["bank_balance"] / salary_safe).fillna(0).clip(upper=50)
    d["emergency_fund_months"] = (
        d["emergency_fund"] / salary_safe.replace(0, np.nan)
    ).fillna(0).clip(upper=24)

    # Estimated EMI for the requested loan at an assumed 12% p.a. flat rate
    rate = 0.12 / 12
    n_t = d["requested_tenure"].clip(lower=1)
    principal = d["requested_amount"]
    d["estimated_emi"] = (
        principal * rate * (1 + rate) ** n_t / ((1 + rate) ** n_t - 1)
    )
    d["requested_dti_ratio"] = (
        (d["current_emi_amount"] + d["estimated_emi"]) / salary_safe
    ).fillna(0)

    # Simple composite risk score (0-100, higher = lower risk)
    credit_component = (d["credit_score"] - 300) / (850 - 300) * 40
    income_component = np.clip(d["affordability_ratio"], -1, 1) * 25 + 25
    stability_component = np.clip(d["years_of_employment"] / 15, 0, 1) * 20
    dependents_penalty = np.clip(d["dependents"] * 2, 0, 15)
    d["risk_score"] = np.clip(
        credit_component + income_component * 0.6 + stability_component - dependents_penalty,
        0,
        100,
    )

    return d


ENGINEERED_COLS = [
    "total_monthly_expenses",
    "disposable_income",
    "debt_to_income_ratio",
    "expense_to_income_ratio",
    "affordability_ratio",
    "savings_ratio",
    "emergency_fund_months",
    "estimated_emi",
    "requested_dti_ratio",
    "risk_score",
]

ALL_NUMERIC_COLS = RAW_NUMERIC_COLS + ENGINEERED_COLS


class EMIPreprocessor:
    """Fits label encoders + a scaler on training data; reused at inference time."""

    def __init__(self):
        self.encoders = {col: LabelEncoder() for col in CATEGORICAL_COLS}
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = add_derived_features(df)
        for col in CATEGORICAL_COLS:
            d[col + "_enc"] = self.encoders[col].fit_transform(d[col].astype(str))
        encoded_cols = [c + "_enc" for c in CATEGORICAL_COLS]
        d[ALL_NUMERIC_COLS] = self.scaler.fit_transform(d[ALL_NUMERIC_COLS])
        self.feature_names_ = ALL_NUMERIC_COLS + encoded_cols
        return d[self.feature_names_]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = add_derived_features(df)
        for col in CATEGORICAL_COLS:
            le = self.encoders[col]
            # Handle unseen categories gracefully by mapping to the first known class
            known = set(le.classes_)
            safe_vals = d[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            d[col + "_enc"] = le.transform(safe_vals)
        encoded_cols = [c + "_enc" for c in CATEGORICAL_COLS]
        d[ALL_NUMERIC_COLS] = self.scaler.transform(d[ALL_NUMERIC_COLS])
        return d[self.feature_names_]
