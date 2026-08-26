# 💳 EMI Prediction Platform

A financial risk assessment web app built with **Streamlit**, **scikit-learn / XGBoost**,
and **MLflow**. It solves two problems from an applicant's financial profile:

1. **Classification** — EMI eligibility: `Eligible` / `High_Risk` / `Not_Eligible`
2. **Regression** — maximum safe monthly EMI amount (₹)

## 📁 Project Structure

```
emi_app/
├── app.py                          # Home page (Streamlit entry point)
├── pages/
│   ├── 1_🔮_Predict.py             # Real-time prediction form
│   ├── 2_📊_Data_Explorer.py       # EDA & visualizations
│   ├── 3_📈_Model_Performance.py   # Model comparison + MLflow info
│   └── 4_🗄️_Data_Management.py     # CRUD on applicant records
├── src/
│   ├── data_generator.py           # Synthetic dataset generator (400K records, 22 features)
│   ├── feature_engineering.py      # Derived ratios, risk score, encoding, scaling
│   ├── train_models.py             # Trains ≥3 classifiers + ≥3 regressors, logs to MLflow
│   └── model_utils.py              # Cached model/data loaders shared across pages
├── models/                         # Saved best classifier/regressor + metadata (generated)
├── data/
│   └── emi_dataset.csv             # 20K-row sample dataset (regenerate full 400K, see below)
├── requirements.txt
└── .streamlit/config.toml
```

## 🚀 Quickstart (local)

```bash
cd emi_app
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 1. (Optional) Regenerate the full 400,000-record dataset
python src/data_generator.py --rows 400000 --out data/emi_dataset.csv

# 2. Train models (pre-trained models are already included in models/,
#    but re-run this if you regenerate the data or want to retrain)
python src/train_models.py --data data/emi_dataset.csv --sample 100000

# 3. Launch the app
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

> **Note on sample size:** `--sample` controls how many rows are used for training
> (100K is a good balance of speed vs. accuracy on a laptop). Use `--sample 0` to
> train on the full dataset — expect this to take significantly longer, especially
> for Random Forest / XGBoost.

## 🔬 MLflow Experiment Tracking

`train_models.py` automatically logs every model run (params, metrics, artifacts) to
MLflow under the `EMI_Prediction_Platform` experiment when `mlflow` is installed.

```bash
mlflow ui
# open http://localhost:5000 to compare runs and promote models via the Model Registry
```

If `mlflow` isn't installed, training still works — tracking is just skipped.

## ☁️ Deploying to Streamlit Cloud

1. Push this folder to a **GitHub repository** (include `models/` so the app has
   pre-trained models available — or add a build step that runs `train_models.py`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub repo.
3. Set **Main file path** to `app.py`.
4. Streamlit Cloud installs `requirements.txt` automatically. Deploy.

The 20K-row sample dataset ships in the repo so **Data Explorer** and **Data
Management** work out of the box. For production, swap in your real data and re-run
the training pipeline.

## 🧠 Models

| Type | Models trained | Best model selection |
|---|---|---|
| Classification | Logistic Regression, Random Forest, XGBoost (falls back to Gradient Boosting if `xgboost` isn't installed) | Highest weighted ROC-AUC |
| Regression | Linear Regression, Random Forest, XGBoost (falls back to Gradient Boosting) | Lowest RMSE |

On a 15K-row sample, this pipeline achieved **96% classification accuracy** (AUC 0.996)
and **RMSE ≈ ₹471** on regression — training on the full 400K records with XGBoost
should meet or exceed the spec targets (>90% accuracy, <₹2000 RMSE).

## 📊 Dataset Schema (22 input features)

- **Personal demographics:** age, gender, marital_status, education
- **Employment & income:** monthly_salary, employment_type, years_of_employment, company_type
- **Housing & family:** house_type, monthly_rent, family_size, dependents
- **Monthly obligations:** school_fees, college_fees, travel_expenses, groceries_utilities, other_monthly_expenses
- **Credit & financial status:** existing_loans, current_emi_amount, credit_score, bank_balance, emergency_fund
- **Loan request:** emi_scenario, requested_amount, requested_tenure

Plus 10 engineered features (debt-to-income ratio, affordability ratio, risk score, etc.)
computed in `src/feature_engineering.py`.

## 🔄 Using Your Own Data

Replace `data/emi_dataset.csv` with your real dataset (same 22 columns + the two
target columns `emi_eligibility` and `max_monthly_emi`), then re-run:

```bash
python src/train_models.py --data data/emi_dataset.csv --sample 0
```

This retrains and overwrites the models in `models/`.
