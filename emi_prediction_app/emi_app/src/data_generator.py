"""
EMI Dataset Generator
======================
Generates a realistic synthetic financial dataset matching the project spec:
- 22 input features (demographics, employment, housing, obligations, credit, loan request)
- 5 EMI scenarios (E-commerce, Home Appliances, Vehicle, Personal Loan, Education)
- 2 targets: emi_eligibility (classification), max_monthly_emi (regression)

Run directly to produce the full 400,000-record dataset:
    python src/data_generator.py --rows 400000 --out data/emi_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd

SCENARIOS = {
    "Ecommerce_Shopping": {"amount": (10_000, 200_000), "tenure": (3, 24)},
    "Home_Appliances": {"amount": (20_000, 300_000), "tenure": (6, 36)},
    "Vehicle": {"amount": (80_000, 1_500_000), "tenure": (12, 84)},
    "Personal_Loan": {"amount": (50_000, 1_000_000), "tenure": (12, 60)},
    "Education": {"amount": (50_000, 500_000), "tenure": (6, 48)},
}

EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]
EMPLOYMENT_TYPES = ["Private", "Government", "Self-employed"]
COMPANY_TYPES = ["Startup", "SME", "MNC", "Government"]
HOUSE_TYPES = ["Rented", "Own", "Family"]


def generate_dataset(n_rows: int = 400_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_per_scenario = n_rows // len(SCENARIOS)
    frames = []

    for scenario, cfg in SCENARIOS.items():
        n = n_per_scenario
        age = rng.integers(25, 61, n)
        gender = rng.choice(["Male", "Female"], n)
        marital_status = rng.choice(["Single", "Married"], n, p=[0.42, 0.58])
        education = rng.choice(EDUCATION_LEVELS, n, p=[0.15, 0.40, 0.30, 0.15])

        # Income tends to rise with education & age
        edu_bonus = pd.Series(education).map(
            {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 2.5}
        ).to_numpy()
        base_salary = 15_000 + edu_bonus * 18_000 + (age - 25) * 900
        monthly_salary = np.clip(
            base_salary + rng.normal(0, 12_000, n), 15_000, 200_000
        ).round(-2)

        employment_type = rng.choice(EMPLOYMENT_TYPES, n, p=[0.55, 0.20, 0.25])
        years_of_employment = np.clip(
            (age - 22) - rng.integers(0, 6, n), 0, 35
        )
        company_type = rng.choice(COMPANY_TYPES, n, p=[0.15, 0.35, 0.35, 0.15])

        house_type = rng.choice(HOUSE_TYPES, n, p=[0.40, 0.35, 0.25])
        monthly_rent = np.where(
            house_type == "Rented",
            np.clip(monthly_salary * rng.uniform(0.12, 0.30, n), 3_000, 60_000),
            0,
        ).round(-2)
        family_size = rng.integers(1, 7, n)
        dependents = np.clip(family_size - 1 - rng.integers(0, 2, n), 0, 5)

        school_fees = np.where(
            dependents > 0, rng.uniform(500, 8_000, n) * (dependents > 0), 0
        ).round(-1)
        college_fees = np.where(
            (dependents > 0) & (age > 40),
            rng.uniform(1_000, 15_000, n),
            0,
        ).round(-1)
        travel_expenses = np.clip(rng.normal(4_000, 2_000, n), 500, 20_000).round(-1)
        groceries_utilities = np.clip(
            monthly_salary * rng.uniform(0.08, 0.20, n), 2_000, 40_000
        ).round(-1)
        other_monthly_expenses = np.clip(
            rng.normal(3_000, 1_500, n), 0, 25_000
        ).round(-1)

        existing_loans = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
        current_emi_amount = np.where(
            existing_loans == "Yes",
            np.clip(monthly_salary * rng.uniform(0.05, 0.35, n), 500, 60_000),
            0,
        ).round(-1)

        credit_score = np.clip(
            rng.normal(650, 90, n) - (existing_loans == "Yes") * rng.integers(0, 40, n),
            300,
            850,
        ).round().astype(int)
        bank_balance = np.clip(
            monthly_salary * rng.uniform(0.3, 4.0, n), 0, 2_000_000
        ).round(-2)
        emergency_fund = np.clip(
            bank_balance * rng.uniform(0.0, 0.6, n), 0, 1_000_000
        ).round(-2)

        emi_scenario = np.full(n, scenario)
        amt_lo, amt_hi = cfg["amount"]
        ten_lo, ten_hi = cfg["tenure"]
        requested_amount = rng.integers(amt_lo, amt_hi + 1, n)
        requested_tenure = rng.integers(ten_lo, ten_hi + 1, n)

        df = pd.DataFrame(
            {
                "age": age,
                "gender": gender,
                "marital_status": marital_status,
                "education": education,
                "monthly_salary": monthly_salary,
                "employment_type": employment_type,
                "years_of_employment": years_of_employment,
                "company_type": company_type,
                "house_type": house_type,
                "monthly_rent": monthly_rent,
                "family_size": family_size,
                "dependents": dependents,
                "school_fees": school_fees,
                "college_fees": college_fees,
                "travel_expenses": travel_expenses,
                "groceries_utilities": groceries_utilities,
                "other_monthly_expenses": other_monthly_expenses,
                "existing_loans": existing_loans,
                "current_emi_amount": current_emi_amount,
                "credit_score": credit_score,
                "bank_balance": bank_balance,
                "emergency_fund": emergency_fund,
                "emi_scenario": emi_scenario,
                "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
            }
        )
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ---- Derive targets from financial capacity ----
    total_expenses = (
        data["monthly_rent"]
        + data["school_fees"]
        + data["college_fees"]
        + data["travel_expenses"]
        + data["groceries_utilities"]
        + data["other_monthly_expenses"]
        + data["current_emi_amount"]
    )
    disposable_income = data["monthly_salary"] - total_expenses
    # Safe EMI = a portion of disposable income, adjusted by credit score & job stability
    credit_factor = (data["credit_score"] - 300) / (850 - 300)
    stability_factor = np.clip(data["years_of_employment"] / 10, 0, 1) * 0.5 + 0.5
    max_monthly_emi = np.clip(
        disposable_income * 0.5 * (0.6 + 0.4 * credit_factor) * stability_factor,
        500,
        50_000,
    )
    noise = np.random.default_rng(seed + 1).normal(0, 400, len(data))
    data["max_monthly_emi"] = np.clip(max_monthly_emi + noise, 500, 50_000).round(-1)

    # Estimated EMI for the requested amount/tenure (flat-rate approx at 12% p.a.)
    rate = 0.12 / 12
    n_t = data["requested_tenure"]
    principal = data["requested_amount"]
    estimated_emi = principal * rate * (1 + rate) ** n_t / ((1 + rate) ** n_t - 1)

    dti_ratio = (data["current_emi_amount"] + estimated_emi) / data["monthly_salary"]

    conditions = [
        (data["max_monthly_emi"] >= estimated_emi * 1.1) & (dti_ratio < 0.45) & (data["credit_score"] >= 650),
        (data["max_monthly_emi"] >= estimated_emi * 0.75) & (dti_ratio < 0.60) & (data["credit_score"] >= 550),
    ]
    choices = ["Eligible", "High_Risk"]
    data["emi_eligibility"] = np.select(conditions, choices, default="Not_Eligible")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the EMI dataset")
    parser.add_argument("--rows", type=int, default=400_000)
    parser.add_argument("--out", type=str, default="data/emi_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_dataset(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} rows -> {args.out}")
    print(df["emi_eligibility"].value_counts(normalize=True))
