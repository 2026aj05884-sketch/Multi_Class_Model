"""Shared schema, input-cleaning and metric helpers for the purchase-intent project.

Both ``train_models.py`` (offline training) and ``app.py`` (Streamlit front-end)
import this module, so the column schema, the cleaning rules applied to raw CSV
rows and the metric definitions are guaranteed to be identical in both places.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "online_shoppers_intention.csv"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
MODEL_DIR = PROJECT_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
TEST_METRICS_PATH = MODEL_DIR / "test_metrics.csv"

UCI_PAGE_URL = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
GITHUB_REPO_URL = "https://github.com/2026aj05884-sketch/ML2"

RANDOM_STATE = 5884

TARGET_COLUMN = "Revenue"
CLASS_NAMES = ["No purchase", "Purchase"]  # position == encoded label

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
CODED_CATEGORICAL_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]
CATEGORICAL_FEATURES = ["Month", *CODED_CATEGORICAL_FEATURES, "VisitorType", "Weekend"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "Administrative": "Number of account/administrative pages visited in the session",
    "Administrative_Duration": "Seconds spent on administrative pages",
    "Informational": "Number of informational pages (e.g. contact, about) visited",
    "Informational_Duration": "Seconds spent on informational pages",
    "ProductRelated": "Number of product pages visited",
    "ProductRelated_Duration": "Seconds spent on product pages",
    "BounceRates": "Average Google Analytics bounce rate of the visited pages",
    "ExitRates": "Average Google Analytics exit rate of the visited pages",
    "PageValues": "Average value of the visited pages before a transaction",
    "SpecialDay": "Closeness of the visit to a special day such as Valentine's Day (0-1)",
    "Month": "Month of the visit",
    "OperatingSystems": "Operating-system code of the visitor",
    "Browser": "Browser code of the visitor",
    "Region": "Geographic region code of the visitor",
    "TrafficType": "Traffic-source code (banner, SMS, direct, ...)",
    "VisitorType": "Returning_Visitor, New_Visitor or Other",
    "Weekend": "Whether the visit happened on a weekend",
    "Revenue": "Target - whether the session ended in a purchase",
}

# Display name -> file inside model/. Order is the order used in the README table.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}

# The source data spells June in full and abbreviates every other month.
_MONTH_CANONICAL = {
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr", "may": "May", "jun": "June",
    "jul": "Jul", "aug": "Aug", "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
}


def to_binary(values: pd.Series) -> pd.Series:
    """Map bool-like values (True/False, TRUE/FALSE, 1/0, yes/no) to 1.0/0.0; anything else -> NaN."""
    if pd.api.types.is_bool_dtype(values) and not values.isna().any():
        return values.astype(float)
    tokens = values.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.nan, index=values.index, dtype=float)
    mapped[tokens.isin(_TRUE_TOKENS)] = 1.0
    mapped[tokens.isin(_FALSE_TOKENS)] = 0.0
    return mapped


def missing_columns(frame: pd.DataFrame, require_target: bool = True) -> list[str]:
    expected = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    return [col for col in expected if col not in frame.columns]


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the model-ready feature frame plus counts of values that had to be repaired.

    Raw CSV rows are cleaned identically at training time and inside the app:
    numeric columns are coerced to numbers and clipped at 0 (every numeric
    feature is a count, duration, rate or page value), categorical columns are
    turned into plain string labels (e.g. Browser 2 -> "2", Weekend TRUE -> "True"),
    and anything unparseable becomes NaN so the imputers inside the saved
    pipelines can deal with it.
    """
    features = pd.DataFrame(index=frame.index)
    issues = {"unparseable_values": 0, "negative_values_clipped": 0}

    def coerce_numeric(col: str) -> pd.Series:
        numeric = pd.to_numeric(frame[col], errors="coerce").astype(float)
        issues["unparseable_values"] += int((numeric.isna() & frame[col].notna()).sum())
        return numeric

    for col in NUMERIC_FEATURES:
        numeric = coerce_numeric(col)
        issues["negative_values_clipped"] += int((numeric < 0).sum())
        features[col] = numeric.clip(lower=0)

    for col in CODED_CATEGORICAL_FEATURES:
        features[col] = coerce_numeric(col).map(lambda code: f"{code:g}", na_action="ignore")

    month = frame["Month"].astype(str).str.strip().str.lower().str[:3].map(_MONTH_CANONICAL)
    issues["unparseable_values"] += int((month.isna() & frame["Month"].notna()).sum())
    features["Month"] = month

    features["VisitorType"] = frame["VisitorType"].astype(str).str.strip().where(frame["VisitorType"].notna())

    weekend = to_binary(frame["Weekend"])
    issues["unparseable_values"] += int((weekend.isna() & frame["Weekend"].notna()).sum())
    features["Weekend"] = weekend.map({1.0: "True", 0.0: "False"})

    # object dtype keeps the encoders' input identical under pandas 2.x and 3.x.
    for col in CATEGORICAL_FEATURES:
        features[col] = features[col].astype(object)
    return features[FEATURE_COLUMNS], issues


def prepare_target(frame: pd.DataFrame) -> pd.Series:
    """Encode the Revenue column as 1.0 (purchase) / 0.0 (no purchase); unparseable -> NaN."""
    return to_binary(frame[TARGET_COLUMN])


def positive_class_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Score used for ROC-AUC: P(purchase) when available, else the decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """The six assignment metrics. Precision / Recall / F1 refer to the Purchase class."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    both_classes_present = np.unique(y_true).size == 2
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)) if both_classes_present else float("nan"),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
