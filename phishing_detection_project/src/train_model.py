"""
train_model.py
---------------
Stage D of the pipeline described in the paper: "Model Training and
Evaluation". Trains Decision Tree, Random Forest, Logistic Regression,
SVM, and (if installed) XGBoost, compares them on accuracy, precision,
recall, F1-score and false-positive rate, then saves the best model
(by F1-score) plus its scaler and metadata to models/.

Usage
-----
    python src/train_model.py
    python src/train_model.py --csv data/phishing_dataset.csv
    python src/train_model.py --real-csv /path/to/PhiUSIIL_dataset.csv --label-col label

If --real-csv is given, that file is used instead of the synthetic
demo dataset. It must contain a numeric label column (0 = legitimate,
1 = phishing) -- point --label-col at it if it isn't named "label".
Non-numeric / identifier columns (like the raw URL) are dropped
automatically.
"""

import argparse
import csv
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import FEATURE_NAMES  # noqa: E402

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_dataset(csv_path, label_col="label", drop_cols=None):
    drop_cols = set(drop_cols or [])
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames:
            raise ValueError(f"CSV file '{csv_path}' is empty or missing a header row.")
        if label_col not in reader.fieldnames:
            raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV file '{csv_path}' contains no data rows.")

    fieldnames = [name for name in reader.fieldnames if name not in drop_cols and name != label_col]
    numeric_fields = []
    for name in fieldnames:
        try:
            float(rows[0][name])
            numeric_fields.append(name)
        except (TypeError, ValueError):
            continue

    X_list = []
    y_list = []
    for row in rows:
        values = []
        for name in numeric_fields:
            raw_value = row.get(name, "")
            if raw_value in (None, ""):
                values.append(0.0)
            else:
                values.append(float(raw_value))
        X_list.append(values)
        y_list.append(int(float(row[label_col])))

    if not X_list or not numeric_fields:
        raise ValueError(f"No numeric feature columns found in '{csv_path}'.")

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=int)
    return X, y, numeric_fields


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def build_candidate_models():
    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        )
    return models


def main():
    parser = argparse.ArgumentParser(description="Train & compare phishing-detection models")
    parser.add_argument("--csv", type=str,
                         default=os.path.join(os.path.dirname(__file__), "..", "data", "phishing_dataset.csv"))
    parser.add_argument("--real-csv", type=str, default=None,
                         help="Path to the real PhiUSIIL (or similar) dataset CSV, if you have it")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    csv_path = args.real_csv if args.real_csv else args.csv
    print(f"Loading dataset: {csv_path}")
    X, y, feature_names = load_dataset(csv_path, label_col=args.label_col, drop_cols=["url"])
    print(f"Dataset shape: {X.shape}, classes: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_candidate_models()
    # These model families need scaled input; tree-based ones don't but
    # scaling doesn't hurt them, so we use scaled features everywhere
    # for a single consistent inference path.
    results = []
    trained = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        elapsed = time.time() - t0

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "false_positive_rate": round(false_positive_rate(y_test, y_pred), 4),
            "train_seconds": round(elapsed, 2),
        }
        results.append(metrics)
        trained[name] = model
        print(f"  {name:22s} acc={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}  "
              f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
              f"fpr={metrics['false_positive_rate']:.4f}  ({elapsed:.2f}s)")

    # Prefer robust ensemble models (RandomForest, XGBoost) when metrics are tied or comparable
    model_pref = {"RandomForest": 5, "XGBoost": 4, "SVM": 3, "LogisticRegression": 2, "DecisionTree": 1}
    results_sorted = sorted(
        results,
        key=lambda r: (r["f1_score"], -r["false_positive_rate"], model_pref.get(r["model"], 0)),
        reverse=True,
    )
    best_name = results_sorted[0]["model"]
    best_model = trained[best_name]
    print(f"\nBest model selected: {best_name} (F1={results_sorted[0]['f1_score']})")

    # Feature importance / coefficients for the explainability layer
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
    importance_ranking = sorted(
        zip(feature_names, importances.tolist()), key=lambda t: t[1], reverse=True
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    metadata = {
        "best_model": best_name,
        "feature_names": feature_names,
        "all_results": results_sorted,
        "feature_importance_ranking": importance_ranking,
        "trained_on": os.path.abspath(csv_path),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: models/best_model.joblib, models/scaler.joblib, models/metadata.json")


if __name__ == "__main__":
    main()
