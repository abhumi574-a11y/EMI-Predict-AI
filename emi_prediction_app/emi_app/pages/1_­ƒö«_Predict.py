import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model_utils import models_available, predict_applicant  # noqa: E402

st.set_page_config(page_title="Predict | EMI Platform", page_icon="🔮", layout="wide")
st.title("🔮 Real-Time EMI Prediction")
st.caption("Enter an applicant's financial profile to get an instant eligibility decision and safe EMI recommendation.")

if not models_available():
    st.error(
        "Trained models not found. Run `python src/train_models.py` from the project "
        "root first, then reload this page."
    )
    st.stop()

with st.form("applicant_form"):
    st.subheader("👤 Personal Demographics")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", 18, 75, 32)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    marital_status = c3.selectbox("Marital Status", ["Single", "Married"])
    education = c4.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

    st.subheader("💼 Employment & Income")
    c1, c2, c3, c4 = st.columns(4)
    monthly_salary = c1.number_input("Monthly Salary (₹)", 15_000, 200_000, 55_000, step=1000)
    employment_type = c2.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    years_of_employment = c3.number_input("Years of Employment", 0, 40, 5)
    company_type = c4.selectbox("Company Type", ["Startup", "SME", "MNC", "Government"])

    st.subheader("🏠 Housing & Family")
    c1, c2, c3, c4 = st.columns(4)
    house_type = c1.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = c2.number_input("Monthly Rent (₹)", 0, 100_000, 8_000, step=500)
    family_size = c3.number_input("Family Size", 1, 12, 3)
    dependents = c4.number_input("Dependents", 0, 10, 1)

    st.subheader("🧾 Monthly Financial Obligations")
    c1, c2, c3, c4, c5 = st.columns(5)
    school_fees = c1.number_input("School Fees (₹)", 0, 30_000, 0, step=500)
    college_fees = c2.number_input("College Fees (₹)", 0, 40_000, 0, step=500)
    travel_expenses = c3.number_input("Travel Expenses (₹)", 0, 30_000, 3_000, step=500)
    groceries_utilities = c4.number_input("Groceries/Utilities (₹)", 0, 60_000, 8_000, step=500)
    other_monthly_expenses = c5.number_input("Other Expenses (₹)", 0, 40_000, 2_000, step=500)

    st.subheader("🏦 Financial Status & Credit History")
    c1, c2, c3, c4, c5 = st.columns(5)
    existing_loans = c1.selectbox("Existing Loans?", ["No", "Yes"])
    current_emi_amount = c2.number_input("Current EMI (₹)", 0, 100_000, 0, step=500)
    credit_score = c3.number_input("Credit Score", 300, 850, 700)
    bank_balance = c4.number_input("Bank Balance (₹)", 0, 5_000_000, 150_000, step=5000)
    emergency_fund = c5.number_input("Emergency Fund (₹)", 0, 2_000_000, 50_000, step=5000)

    st.subheader("📝 Loan Application Details")
    c1, c2, c3 = st.columns(3)
    emi_scenario = c1.selectbox(
        "EMI Scenario",
        ["Ecommerce_Shopping", "Home_Appliances", "Vehicle", "Personal_Loan", "Education"],
    )
    requested_amount = c2.number_input("Requested Amount (₹)", 5_000, 2_000_000, 150_000, step=5000)
    requested_tenure = c3.number_input("Requested Tenure (months)", 3, 96, 24)

    submitted = st.form_submit_button("🚀 Predict", use_container_width=True, type="primary")

if submitted:
    applicant = dict(
        age=age, gender=gender, marital_status=marital_status, education=education,
        monthly_salary=monthly_salary, employment_type=employment_type,
        years_of_employment=years_of_employment, company_type=company_type,
        house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
        dependents=dependents, school_fees=school_fees, college_fees=college_fees,
        travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
        other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
        current_emi_amount=current_emi_amount, credit_score=credit_score,
        bank_balance=bank_balance, emergency_fund=emergency_fund,
        emi_scenario=emi_scenario, requested_amount=requested_amount,
        requested_tenure=requested_tenure,
    )

    eligibility, proba_dict, max_emi = predict_applicant(applicant)

    st.divider()
    st.subheader("📋 Prediction Results")

    color_map = {"Eligible": "green", "High_Risk": "orange", "Not_Eligible": "red"}
    icon_map = {"Eligible": "✅", "High_Risk": "⚠️", "Not_Eligible": "❌"}

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"### {icon_map[eligibility]} Eligibility: :{color_map[eligibility]}[{eligibility}]")
        proba_df = pd.DataFrame(
            {"Class": list(proba_dict.keys()), "Probability": list(proba_dict.values())}
        ).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Class"))
        st.dataframe(
            proba_df.assign(Probability=lambda d: (d["Probability"] * 100).round(1).astype(str) + "%"),
            hide_index=True, use_container_width=True,
        )

    with r2:
        st.markdown("### 💰 Maximum Safe Monthly EMI")
        st.metric("Recommended max EMI", f"₹{max_emi:,.0f}")

        rate = 0.12 / 12
        n_t = requested_tenure
        principal = requested_amount
        estimated_emi = principal * rate * (1 + rate) ** n_t / ((1 + rate) ** n_t - 1)
        st.metric("Estimated EMI for this request (@12% p.a.)", f"₹{estimated_emi:,.0f}")

        if estimated_emi <= max_emi:
            st.success(
                f"The requested EMI (₹{estimated_emi:,.0f}) is within the applicant's "
                f"safe capacity (₹{max_emi:,.0f}). ✅"
            )
        else:
            st.warning(
                f"The requested EMI (₹{estimated_emi:,.0f}) **exceeds** the safe capacity "
                f"(₹{max_emi:,.0f}) by ₹{estimated_emi - max_emi:,.0f}/month. Consider a "
                f"longer tenure or lower loan amount."
            )

    with st.expander("🔍 View computed financial ratios"):
        total_exp = (monthly_rent + school_fees + college_fees + travel_expenses
                     + groceries_utilities + other_monthly_expenses + current_emi_amount)
        dti = current_emi_amount / monthly_salary if monthly_salary else 0
        exp_ratio = total_exp / monthly_salary if monthly_salary else 0
        st.write(f"- **Total monthly expenses:** ₹{total_exp:,.0f}")
        st.write(f"- **Disposable income:** ₹{monthly_salary - total_exp:,.0f}")
        st.write(f"- **Debt-to-income ratio:** {dti:.1%}")
        st.write(f"- **Expense-to-income ratio:** {exp_ratio:.1%}")
