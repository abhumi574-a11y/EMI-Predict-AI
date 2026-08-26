import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model_utils import DATA_DIR, data_available  # noqa: E402

st.set_page_config(page_title="Data Management | EMI Platform", page_icon="🗄️", layout="wide")
st.title("🗄️ Data Management")
st.caption("Create, read, update, and delete applicant records. Changes apply to your in-session working copy.")

if not data_available():
    st.error("No dataset found at `data/emi_dataset.csv`. Generate it first from `src/data_generator.py`.")
    st.stop()

if "working_df" not in st.session_state:
    full = pd.read_csv(os.path.join(DATA_DIR, "emi_dataset.csv"))
    st.session_state.working_df = full.sample(n=min(2000, len(full)), random_state=1).reset_index(drop=True)

df = st.session_state.working_df

st.caption(
    f"Working copy: **{len(df):,} records** (a session sample — not the full 400K "
    "dataset, kept small for responsive editing)."
)

tab_view, tab_add, tab_edit, tab_delete = st.tabs(["👁️ View / Search", "➕ Add Record", "✏️ Edit Table", "🗑️ Delete"])

with tab_view:
    c1, c2, c3 = st.columns(3)
    scenario_filter = c1.multiselect("Filter by EMI Scenario", sorted(df["emi_scenario"].unique()))
    eligibility_filter = c2.multiselect("Filter by Eligibility", sorted(df["emi_eligibility"].unique()))
    min_credit = c3.slider("Minimum Credit Score", 300, 850, 300)

    filtered = df.copy()
    if scenario_filter:
        filtered = filtered[filtered["emi_scenario"].isin(scenario_filter)]
    if eligibility_filter:
        filtered = filtered[filtered["emi_eligibility"].isin(eligibility_filter)]
    filtered = filtered[filtered["credit_score"] >= min_credit]

    st.write(f"**{len(filtered):,}** matching records")
    st.dataframe(filtered, use_container_width=True, height=400)
    st.download_button(
        "⬇️ Export filtered records as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="emi_records_filtered.csv",
        mime="text/csv",
    )

with tab_add:
    st.subheader("Add a new applicant record")
    with st.form("add_record_form"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 18, 75, 30)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        marital_status = c3.selectbox("Marital Status", ["Single", "Married"])
        education = c4.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

        c1, c2, c3, c4 = st.columns(4)
        monthly_salary = c1.number_input("Monthly Salary", 15_000, 200_000, 50_000, step=1000)
        employment_type = c2.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = c3.number_input("Years of Employment", 0, 40, 3)
        company_type = c4.selectbox("Company Type", ["Startup", "SME", "MNC", "Government"])

        c1, c2, c3 = st.columns(3)
        credit_score = c1.number_input("Credit Score", 300, 850, 680)
        emi_scenario = c2.selectbox(
            "EMI Scenario",
            ["Ecommerce_Shopping", "Home_Appliances", "Vehicle", "Personal_Loan", "Education"],
        )
        requested_amount = c3.number_input("Requested Amount", 5_000, 2_000_000, 100_000, step=5000)

        add_submitted = st.form_submit_button("➕ Add Record", type="primary")

    if add_submitted:
        new_row = {col: (df[col].mode()[0] if df[col].dtype == object else df[col].median()) for col in df.columns}
        new_row.update(
            age=age, gender=gender, marital_status=marital_status, education=education,
            monthly_salary=monthly_salary, employment_type=employment_type,
            years_of_employment=years_of_employment, company_type=company_type,
            credit_score=credit_score, emi_scenario=emi_scenario,
            requested_amount=requested_amount,
        )
        st.session_state.working_df = pd.concat(
            [df, pd.DataFrame([new_row])], ignore_index=True
        )
        st.success("Record added to the working copy. See it in **View / Search**.")
        st.rerun()

with tab_edit:
    st.subheader("Edit records directly in the table")
    st.caption("Double-click a cell to edit. Use the ⋮ menu on a row to delete it. Click 'Save Changes' when done.")
    edited = st.data_editor(
        df, use_container_width=True, height=450, num_rows="dynamic", key="editor"
    )
    if st.button("💾 Save Changes", type="primary"):
        st.session_state.working_df = edited.reset_index(drop=True)
        st.success("Changes saved to the working copy.")
        st.rerun()

with tab_delete:
    st.subheader("Delete records by index")
    idx_to_delete = st.multiselect("Select row indices to delete", df.index.tolist())
    if st.button("🗑️ Delete Selected", type="secondary"):
        st.session_state.working_df = df.drop(index=idx_to_delete).reset_index(drop=True)
        st.success(f"Deleted {len(idx_to_delete)} record(s).")
        st.rerun()

    st.divider()
    if st.button("♻️ Reset working copy to original sample"):
        full = pd.read_csv(os.path.join(DATA_DIR, "emi_dataset.csv"))
        st.session_state.working_df = full.sample(n=min(2000, len(full)), random_state=1).reset_index(drop=True)
        st.success("Working copy reset.")
        st.rerun()
