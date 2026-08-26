"""
Model Training Pipeline
=========================
Trains >= 3 classification models (EMI eligibility) and >= 3 regression models
(max monthly EMI), logs every run to MLflow, and saves the best model of each
type (by ROC-AUC / RMSE) to models/ for the Streamlit app to load.

Usage:
    python src/train_models.py --data data/emi_dataset.csv --sample 100000
"""

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from feature_engineering import EMIPreprocessor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import mlflow
    import mlflow.sklearn

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def get_classification_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=120, max_depth=12, n_jobs=-1, random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoostClassifier"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.15,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
        )
    else:
        models["GradientBoostingClassifier"] = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, random_state=42
        )
    return models


def get_regression_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120, max_depth=12, n_jobs=-1, random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.15,
            n_jobs=-1,
            random_state=42,
        )
    else:
        models["GradientBoostingRegressor"] = GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=42
        )
    return models


def train_classification(X_train, X_test, y_train, y_test, class_labels):
    lb = LabelBinarizer().fit(class_labels)
    results = {}
    best_name, best_model, best_score = None, None, -np.inf

    for name, model in get_classification_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(lb.transform(y_test), proba, multi_class="ovr", average="weighted")
        except ValueError:
            auc = np.nan

        elapsed = time.time() - t0
        metrics = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "roc_auc": auc, "train_time_sec": elapsed,
        }
        results[name] = metrics
        print(f"[Classification] {name}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f} ({elapsed:.1f}s)")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"clf_{name}"):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
                mlflow.sklearn.log_model(model, "model")

        score = auc if not np.isnan(auc) else acc
        if score > best_score:
            best_name, best_model, best_score = name, model, score

    return best_name, best_model, results


def train_regression(X_train, X_test, y_train, y_test):
    results = {}
    best_name, best_model, best_score = None, None, np.inf  # lower RMSE is better

    for name, model in get_regression_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)

        elapsed = time.time() - t0
        metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "train_time_sec": elapsed}
        results[name] = metrics
        print(f"[Regression] {name}: rmse={rmse:.1f} mae={mae:.1f} r2={r2:.4f} ({elapsed:.1f}s)")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"reg_{name}"):
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")

        if rmse < best_score:
            best_name, best_model, best_score = name, model, rmse

    return best_name, best_model, results


def main(data_path: str, sample: int, out_dir: str):
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    print(f"Using {len(df):,} rows for training")

    if HAS_MLFLOW:
        mlflow.set_experiment("EMI_Prediction_Platform")
        print("MLflow experiment tracking enabled -> run `mlflow ui` to view the dashboard")
    else:
        print("MLflow not installed - skipping experiment tracking (pip install mlflow to enable)")

    y_clf = df["emi_eligibility"]
    y_reg = df["max_monthly_emi"]
    X_raw = df.drop(columns=["emi_eligibility", "max_monthly_emi"])

    pre = EMIPreprocessor()
    X = pre.fit_transform(X_raw)

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    print("\n=== Training classification models (EMI eligibility) ===")
    best_clf_name, best_clf, clf_results = train_classification(
        X_train, X_test, yc_train, yc_test, sorted(y_clf.unique())
    )
    print(f"Best classification model: {best_clf_name}")

    print("\n=== Training regression models (max monthly EMI) ===")
    best_reg_name, best_reg, reg_results = train_regression(
        X_train, X_test, yr_train, yr_test
    )
    print(f"Best regression model: {best_reg_name}")

    # ---- Save artifacts ----
    joblib.dump(pre, f"{out_dir}/preprocessor.joblib", compress=3)
    joblib.dump(best_clf, f"{out_dir}/best_classifier.joblib", compress=3)
    joblib.dump(best_reg, f"{out_dir}/best_regressor.joblib", compress=3)

    meta = {
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
        "classification_results": clf_results,
        "regression_results": reg_results,
        "n_rows_trained_on": len(df),
        "feature_names": pre.feature_names_,
        "class_labels": sorted(y_clf.unique().tolist()),
        "used_xgboost": HAS_XGB,
    }
    with open(f"{out_dir}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)

    print(f"\nSaved preprocessor + best models + metadata to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/emi_dataset.csv")
    parser.add_argument("--sample", type=int, default=100_000,
                         help="Rows to sample for training (use 0 for full dataset)")
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()
    main(args.data, args.sample, args.out)
